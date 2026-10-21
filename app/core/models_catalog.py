"""
Canonical Groq model catalog - single source of truth for model metadata.

Every field here was verified against https://console.groq.com/docs/models and
related capability pages (tool-use, vision, structured-outputs) rather than
estimated. Where Groq's docs don't publish a number (e.g. exact param count),
the field is left None rather than guessed.

Both the backend (agent.py tool routing) and the frontend (model selector UI)
read from MODEL_CATALOG so capability/pricing data is defined exactly once.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    id: str                      # exact Groq API model id
    name: str                    # display name
    provider: str                # underlying model creator
    tier: str                    # "production" | "preview"
    description: str             # 1-2 sentence summary
    strengths: list[str]
    weaknesses: list[str]
    best_for: list[str]
    intelligence: int            # 1-5 relative scale within this catalog
    speed: int                   # 1-5 relative scale within this catalog
    cost_tier: str                # "cheap" | "medium" | "expensive"
    input_price_per_m: Optional[float]   # USD per 1M input tokens
    output_price_per_m: Optional[float]  # USD per 1M output tokens
    context_window: Optional[int]        # tokens
    max_output_tokens: Optional[int]
    supports_tool_use: bool
    supports_json_mode: bool
    supports_json_schema: bool   # strict structured outputs
    supports_vision: bool
    supports_documents: bool     # native document/file input
    long_context: bool           # >= 128k
    categories: list[str] = field(default_factory=list)
    color: str = "gray"          # UI accent: purple/blue/green/amber/red/gray


MODEL_CATALOG: dict[str, ModelSpec] = {

    "openai/gpt-oss-120b": ModelSpec(
        id="openai/gpt-oss-120b",
        name="GPT-OSS 120B",
        provider="OpenAI",
        tier="production",
        description="OpenAI's flagship open-weight reasoning model. Currently the "
                     "platform's default and strongest general-purpose model.",
        strengths=["Strong reasoning", "Reliable tool use", "Strict JSON schema support",
                   "Good code generation"],
        weaknesses=["Higher latency than smaller models", "No vision support"],
        best_for=["Complex agents", "Multi-step reasoning", "Production tool-calling",
                   "Code generation"],
        intelligence=5, speed=3, cost_tier="cheap",
        input_price_per_m=0.15, output_price_per_m=0.75,
        context_window=131072, max_output_tokens=65536,
        supports_tool_use=True, supports_json_mode=True, supports_json_schema=True,
        supports_vision=False, supports_documents=False, long_context=True,
        categories=["reasoning", "coding", "agents", "best_quality", "documents"],
        color="purple",
    ),

    "openai/gpt-oss-20b": ModelSpec(
        id="openai/gpt-oss-20b",
        name="GPT-OSS 20B",
        provider="OpenAI",
        tier="production",
        description="Smaller sibling of GPT-OSS 120B - same strict JSON/tool-use "
                     "support at much lower cost and latency.",
        strengths=["Very cheap", "Strict JSON schema support", "Low latency",
                   "Reliable tool use"],
        weaknesses=["Less capable than 120B on hard reasoning tasks"],
        best_for=["High-volume agents", "Cost-sensitive tool calling", "Chat assistants"],
        intelligence=3, speed=5, cost_tier="cheap",
        input_price_per_m=0.075, output_price_per_m=0.30,
        context_window=131072, max_output_tokens=65536,
        supports_tool_use=True, supports_json_mode=True, supports_json_schema=True,
        supports_vision=False, supports_documents=False, long_context=True,
        categories=["communication", "agents", "cheapest", "fastest"],
        color="blue",
    ),

    "llama-3.3-70b-versatile": ModelSpec(
        id="llama-3.3-70b-versatile",
        name="Llama 3.3 70B Versatile",
        provider="Meta",
        tier="production",
        description="Meta's general-purpose 70B model. Solid all-rounder for "
                     "chat and content generation.",
        strengths=["Balanced quality/cost", "Good multilingual support",
                   "Well-tested, mature model"],
        weaknesses=["No strict JSON schema mode (best-effort JSON only)",
                     "No vision support"],
        best_for=["General chat", "Content writing", "Summarization"],
        intelligence=4, speed=3, cost_tier="medium",
        input_price_per_m=0.59, output_price_per_m=0.79,
        context_window=131072, max_output_tokens=32768,
        supports_tool_use=True, supports_json_mode=True, supports_json_schema=False,
        supports_vision=False, supports_documents=False, long_context=True,
        categories=["communication", "reasoning"],
        color="green",
    ),

    "llama-3.1-8b-instant": ModelSpec(
        id="llama-3.1-8b-instant",
        name="Llama 3.1 8B Instant",
        provider="Meta",
        tier="production",
        description="Small, extremely fast Meta model for latency-critical or "
                     "high-volume simple tasks.",
        strengths=["Fastest model in the catalog", "Very cheap"],
        weaknesses=["Limited reasoning depth", "No strict JSON schema mode"],
        best_for=["Real-time chat", "Simple classification", "High-throughput pipelines"],
        intelligence=2, speed=5, cost_tier="cheap",
        input_price_per_m=0.05, output_price_per_m=0.08,
        context_window=131072, max_output_tokens=131072,
        supports_tool_use=True, supports_json_mode=True, supports_json_schema=False,
        supports_vision=False, supports_documents=False, long_context=True,
        categories=["fastest", "cheapest", "voice_agents"],
        color="amber",
    ),

    "qwen/qwen3.6-27b": ModelSpec(
        id="qwen/qwen3.6-27b",
        name="Qwen 3.6 27B",
        provider="Alibaba",
        tier="preview",
        description="The only vision-capable model currently on Groq. Also "
                     "supports tool use. Preview tier - can change or be removed "
                     "by Groq without notice.",
        strengths=["Only vision-capable option available", "Supports tool use",
                   "Strong multilingual performance"],
        weaknesses=["Preview tier (not guaranteed stable)", "No strict JSON schema mode"],
        best_for=["Image understanding", "Document-with-images analysis",
                   "Multilingual tasks"],
        intelligence=4, speed=3, cost_tier="medium",
        input_price_per_m=None, output_price_per_m=None,
        context_window=131072, max_output_tokens=None,
        supports_tool_use=True, supports_json_mode=True, supports_json_schema=False,
        supports_vision=True, supports_documents=True, long_context=True,
        categories=["images", "documents", "analytics"],
        color="pink",
    ),

    "whisper-large-v3": ModelSpec(
        id="whisper-large-v3",
        name="Whisper Large v3",
        provider="OpenAI",
        tier="production",
        description="Speech-to-text transcription model. Not a chat model - "
                     "used for voice agent pipelines.",
        strengths=["High transcription accuracy", "Broad language support"],
        weaknesses=["Not usable in text chat - audio input only"],
        best_for=["Voice agent transcription", "Meeting/call transcription"],
        intelligence=4, speed=4, cost_tier="cheap",
        input_price_per_m=None, output_price_per_m=None,
        context_window=None, max_output_tokens=None,
        supports_tool_use=False, supports_json_mode=False, supports_json_schema=False,
        supports_vision=False, supports_documents=False, long_context=False,
        categories=["voice_agents"],
        color="gray",
    ),

    "playai-tts": ModelSpec(
        id="playai-tts",
        name="PlayAI TTS",
        provider="PlayAI",
        tier="preview",
        description="Text-to-speech synthesis model. Not a chat model - used "
                     "for voice agent pipelines.",
        strengths=["Natural-sounding voice output"],
        weaknesses=["Not usable in text chat - audio output only",
                     "Preview tier"],
        best_for=["Voice agent responses", "Audio content generation"],
        intelligence=3, speed=4, cost_tier="medium",
        input_price_per_m=None, output_price_per_m=None,
        context_window=None, max_output_tokens=None,
        supports_tool_use=False, supports_json_mode=False, supports_json_schema=False,
        supports_vision=False, supports_documents=False, long_context=False,
        categories=["voice_agents"],
        color="gray",
    ),
}

DEFAULT_MODEL_ID = "openai/gpt-oss-120b"

CATEGORIES = [
    {"id": "communication", "label": "Communication", "icon": "\U0001F4AC"},
    {"id": "coding", "label": "Programming", "icon": "\U0001F4BB"},
    {"id": "documents", "label": "Working with Documents", "icon": "\U0001F4C4"},
    {"id": "reasoning", "label": "Logical Reasoning", "icon": "\U0001F9E0"},
    {"id": "analytics", "label": "Analytics", "icon": "\U0001F4CA"},
    {"id": "images", "label": "Working with Images", "icon": "\U0001F5BC"},
    {"id": "voice_agents", "label": "Voice Agents", "icon": "\U0001F3A4"},
    {"id": "fastest", "label": "Maximum Speed", "icon": "\U0001F680"},
    {"id": "cheapest", "label": "Cheapest", "icon": "\U0001F4B0"},
    {"id": "best_quality", "label": "Best Quality", "icon": "\u2B50"},
    {"id": "agents", "label": "AI Agents", "icon": "\U0001F916"},
]


def chat_capable_models() -> dict[str, ModelSpec]:
    """Models usable in the text chat pipeline (excludes audio-only models)."""
    return {k: v for k, v in MODEL_CATALOG.items() if k not in ("whisper-large-v3", "playai-tts")}


def pick_auto_model(task_hint: str, needs_vision: bool = False,
                     needs_json_schema: bool = False, prioritize_speed: bool = False,
                     prioritize_cost: bool = False) -> str:
    """Auto mode: pick the best chat-capable model for a task's actual requirements.
    Hard constraints (vision / strict JSON) filter first; among the remainder we rank
    by intelligence unless the caller explicitly asked to prioritize speed or cost.
    """
    candidates = chat_capable_models()
    if needs_vision:
        candidates = {k: v for k, v in candidates.items() if v.supports_vision}
    if needs_json_schema:
        candidates = {k: v for k, v in candidates.items() if v.supports_json_schema}
    if not candidates:
        return DEFAULT_MODEL_ID

    if prioritize_speed:
        return max(candidates.items(), key=lambda kv: kv[1].speed)[0]
    if prioritize_cost:
        return min(
            candidates.items(),
            key=lambda kv: (kv[1].input_price_per_m if kv[1].input_price_per_m is not None else 999),
        )[0]
    return max(candidates.items(), key=lambda kv: kv[1].intelligence)[0]


def to_dict(spec: ModelSpec) -> dict:
    return {
        "id": spec.id, "name": spec.name, "provider": spec.provider, "tier": spec.tier,
        "description": spec.description, "strengths": spec.strengths,
        "weaknesses": spec.weaknesses, "best_for": spec.best_for,
        "intelligence": spec.intelligence, "speed": spec.speed, "cost_tier": spec.cost_tier,
        "input_price_per_m": spec.input_price_per_m, "output_price_per_m": spec.output_price_per_m,
        "context_window": spec.context_window, "max_output_tokens": spec.max_output_tokens,
        "supports_tool_use": spec.supports_tool_use, "supports_json_mode": spec.supports_json_mode,
        "supports_json_schema": spec.supports_json_schema, "supports_vision": spec.supports_vision,
        "supports_documents": spec.supports_documents, "long_context": spec.long_context,
        "categories": spec.categories, "color": spec.color,
    }
