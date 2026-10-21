"""DatasetAdvisor - looks at a dataset and suggests what's actually worth modeling, instead
of requiring the user to already know (which defeats the point of a no-code AutoML product
for someone without an ML background - see the "C" feature discussed with the product owner:
most of the barrier for a beginner with a real idea isn't writing code, it's knowing what
question to even ask of their data).

DatasetProfiler.suggest_task() already existed before this file - it's a pure heuristic (last
numeric column as target, cardinality-based task type) with zero semantic understanding of
what the columns actually MEAN. "pick the last numeric column" can land on something
meaningless (a customer_id, a timestamp) just as easily as something useful. This module adds
an LLM-powered layer on top of that SAME profiler output: real column stats (free, already
computed, no LLM cost) interpreted semantically into a ranked list of plausible, named
business questions ("predict churn", "estimate price"), not just a column name.

Falls back to the plain profiler heuristic if the LLM call fails for any reason (no key,
provider down, bad JSON back) - this must never be a hard failure, since the point is to help
a beginner get UNSTUCK, not add a new way for them to get stuck.
"""
import json
from typing import Dict, List, Optional

from app.core.engine.profiler import DatasetProfiler


SYSTEM_PROMPT = """You are a senior data scientist helping a complete beginner who has a CSV \
file but doesn't yet know what model to build from it. You'll be given real column \
statistics (not the raw data) from an automated profiler. Suggest 1 to 3 concrete, plausible \
modeling tasks this dataset could support, ranked best-first.

For EACH suggestion give:
- "target_column": an EXACT column name from the ones given (never invent one)
- "task": one of "binary_classification", "multiclass_classification", "regression"
- "title": a short, plain-language name for what this would predict/solve, written for \
someone with no ML background (e.g. "Predict which customers will cancel" not "binary \
classification on churn_flag")
- "description": one or two sentences a complete beginner would understand, explaining what \
building this model would let them DO - not restating the stats
- "confidence": your honest 0-1 confidence this is a sensible, useful target given the column \
names and stats - a column that is obviously an id, a name, or a free-text field is NEVER a \
sensible target even if it's numeric; say so plainly in "description" if the best you can \
offer is low-confidence or the dataset doesn't clearly support any good question

Respond with ONLY a JSON object: {"suggestions": [{"target_column": "...", "task": "...", \
"title": "...", "description": "...", "confidence": 0.0}]}, ordered best-first. If NOTHING \
in the dataset looks like a sensible modeling target, return an empty "suggestions" list \
rather than forcing a bad suggestion - explain why in a top-level "note" field instead."""


def _profile_to_prompt(profile) -> str:
    lines = [f"Dataset: {profile.file_name} - {profile.rows} rows, {profile.columns} columns.", "Columns:"]
    for name, info in profile.column_info.items():
        bits = [f"dtype={info['dtype']}", f"unique={info['unique']} ({info['unique_pct']}%)",
                f"missing={info['missing_pct']}%", f"looks_like={info['suggested_type']}"]
        if "top_values" in info:
            top = ", ".join(f"{k!r}" for k in list(info["top_values"].keys())[:5])
            bits.append(f"sample_values=[{top}]")
        elif info.get("min") is not None:
            bits.append(f"range=[{info['min']}, {info['max']}]")
        lines.append(f"  - {name}: {'; '.join(bits)}")
    return "\n".join(lines)


def suggest_model_ideas(dataset_path: str, groq_key: str = "", model: str = "openai/gpt-oss-120b") -> Dict:
    """Profiles the dataset (free, local, no LLM) and asks the LLM to interpret it into a
    ranked list of plausible modeling tasks. Returns:
      {"success": bool, "suggestions": [...], "note": str|None, "ai_generated": bool,
       "profile_summary": {...}}
    "ai_generated" is False when this fell back to the plain profiler heuristic (no key,
    provider failure, or an unparseable response) - the UI should say so rather than present
    a heuristic guess with the same confidence as a real analysis.
    """
    profiler = DatasetProfiler()
    try:
        profile = profiler.run_full_profile(dataset_path)
    except Exception as e:
        return {"success": False, "error": f"Could not read dataset: {e}", "suggestions": []}

    fallback_task, fallback_target, fallback_conf = profiler.suggest_task()
    fallback_suggestions = [{
        "target_column": fallback_target, "task": fallback_task,
        "title": f"Predict {fallback_target}",
        "description": "Heuristic guess only (picked the last/first numeric-ish column) - "
                        "the AI-powered analysis wasn't available, so this hasn't actually "
                        "looked at what your columns mean.",
        "confidence": fallback_conf,
    }]

    if not groq_key:
        return {"success": True, "suggestions": fallback_suggestions, "note": None,
                "ai_generated": False, "profile_summary": profiler.get_summary()}

    try:
        from app.core.groq_client import call_llm
        completion = call_llm(
            groq_key, model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": _profile_to_prompt(profiler.profile)}],
            temperature=0.2, max_tokens=900, response_format={"type": "json_object"},
        )
        parsed = json.loads(completion.choices[0].message.content)
        suggestions = parsed.get("suggestions", [])
        valid_columns = set(profiler.profile.column_info.keys())
        # Never trust an invented column name, even from a well-formed JSON response - a
        # hallucinated target_column would fail loudly later inside the actual training run
        # instead of here, where it's cheap and clear to just drop the bad suggestion.
        suggestions = [s for s in suggestions if s.get("target_column") in valid_columns]
        if not suggestions:
            return {"success": True, "suggestions": fallback_suggestions,
                    "note": parsed.get("note"), "ai_generated": False,
                    "profile_summary": profiler.get_summary()}
        return {"success": True, "suggestions": suggestions, "note": parsed.get("note"),
                "ai_generated": True, "profile_summary": profiler.get_summary()}
    except Exception as e:
        print(f"[DatasetAdvisor] AI analysis failed ({type(e).__name__}: {e}) - falling back "
              f"to the plain heuristic.")
        return {"success": True, "suggestions": fallback_suggestions, "note": None,
                "ai_generated": False, "profile_summary": profiler.get_summary()}
