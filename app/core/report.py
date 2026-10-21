"""One-file client report for a forecast: HTML with an inline SVG chart, no external resources.

    build_forecast_report(df, ...) -> {"html": str, "summary": {...}}

Design rules (the reason this exists - the product's difference is that it says when NOT to trust it):
- every number comes from the engine's own held-out evaluation (Forecaster.run), nothing is invented here;
- the "80%" range is shown together with the coverage MEASURED on held-out data, not as a promise;
- the verdict sentence follows the engine's trust level; a "low" verdict says so in plain words;
- everything that comes from the user's file (column names, client name, notes) is HTML-escaped;
- the page has no scripts, no remote fonts, no images: it opens offline and prints cleanly ("Save as PDF").
"""
from __future__ import annotations

import html
import re
import tempfile
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.engine.transformers import _infer_dayfirst, _parse

_DATE_HINTS = ("date", "day", "period", "month", "week", "time", "дата", "день", "период", "месяц", "неделя")
_TARGET_HINTS = ("sales", "revenue", "amount", "units", "qty", "quantity", "orders", "total", "income",
                 "продаж", "выруч", "сумм", "кол", "заказ", "доход", "оборот")

_T = {
    "en": {
        "title": "Forecast report", "for": "Prepared for", "target": "Forecasting", "horizon": "Horizon",
        "periods": "periods", "history": "History used", "points": "data points", "verdict": "How much to trust this",
        "level": {"high": "High reliability", "medium": "Medium reliability", "low": "Low reliability"},
        "sentence": {
            "high": "On data the model never saw, it clearly beat the simple baselines. You can plan with this forecast, within the stated range.",
            "medium": "The model beat the simple baselines, but only modestly or with limited history. Use it as guidance, not as a precise figure.",
            "low": "The model did not clearly beat the simplest forecasts, or the series has no predictable pattern. Do not rely on the numbers below.",
        },
        "same": "This forecast is essentially a simple rule ({m}): the model found nothing reliably better. It is a sensible baseline, not an insight. Use it as guidance only.",
        "forecast": "Forecast", "date": "Date", "value": "Forecast", "low_c": "Range from", "high_c": "Range to",
        "chart": "Recent history and forecast", "history_l": "History", "forecast_l": "Forecast", "range_l": "Likely range",
        "evidence": "The evidence", "evidence_intro": "Tested on {n} rolling forecasts of {h} periods each, using only data the model had not been trained on. MAE is the average absolute error in the units of your data; lower is better.",
        "method": "Method", "mae": "Average error (MAE)", "model": "This model", "seasonal": "Same period last cycle",
        "last": "Last known value", "mavg": "Moving average", "mean": "Overall average",
        "better": "Error is {p:.0f}% lower than the best simple method.", "worse": "Error is {p:.0f}% higher than the best simple method.",
        "cov_close": "The range is meant to contain the real value 8 times out of 10. On held-out data it did so {c:.0%} of the time: close to the target.",
        "cov_low": "The range is meant to contain the real value 8 times out of 10. On held-out data it did so only {c:.0%} of the time, so treat it as an approximate, slightly optimistic range.",
        "limits": "Limits", "limit_list": [
            "The model only sees the history in your file. Promotions, holidays, price changes, stock-outs and other events are unknown to it.",
            "It assumes the future behaves like the past. Forecasts further out are less reliable than the next few periods.",
            "This is a statistical estimate, not a guarantee.",
        ],
        "notes": "Data notes", "gen": "Generated on {d} by GenesisAI", "chosen": "Chosen method",
    },
    "ru": {
        "title": "Отчёт по прогнозу", "for": "Подготовлено для", "target": "Прогнозируем", "horizon": "Горизонт",
        "periods": "периодов", "history": "Использована история", "points": "точек данных", "verdict": "Насколько этому можно верить",
        "level": {"high": "Высокая надёжность", "medium": "Средняя надёжность", "low": "Низкая надёжность"},
        "sentence": {
            "high": "На данных, которых модель не видела, она заметно обошла простые методы. Прогнозом можно пользоваться при планировании в пределах указанного диапазона.",
            "medium": "Модель обошла простые методы, но ненамного или на небольшой истории. Используйте прогноз как ориентир, а не как точную цифру.",
            "low": "Модель не обошла простейшие прогнозы или в ряде нет предсказуемых закономерностей. Не опирайтесь на цифры ниже.",
        },
        "same": "Этот прогноз по сути простое правило ({m}): модель не нашла ничего надёжно лучше. Это разумная опорная оценка, а не открытие. Используйте только как ориентир.",
        "forecast": "Прогноз", "date": "Дата", "value": "Прогноз", "low_c": "Диапазон от", "high_c": "Диапазон до",
        "chart": "Недавняя история и прогноз", "history_l": "История", "forecast_l": "Прогноз", "range_l": "Вероятный диапазон",
        "evidence": "Доказательства", "evidence_intro": "Проверено на {n} скользящих прогнозах по {h} периодов, только на данных, на которых модель не обучалась. MAE — средняя абсолютная ошибка в единицах ваших данных, чем меньше, тем лучше.",
        "method": "Метод", "mae": "Средняя ошибка (MAE)", "model": "Эта модель", "seasonal": "Как в тот же период прошлого цикла",
        "last": "Последнее известное значение", "mavg": "Скользящее среднее", "mean": "Общее среднее",
        "better": "Ошибка на {p:.0f}% ниже, чем у лучшего простого метода.", "worse": "Ошибка на {p:.0f}% выше, чем у лучшего простого метода.",
        "cov_close": "Диапазон должен содержать реальное значение в 8 случаях из 10. На отложенных данных так и было в {c:.0%} случаев: близко к цели.",
        "cov_low": "Диапазон должен содержать реальное значение в 8 случаях из 10. На отложенных данных так было лишь в {c:.0%} случаев, поэтому считайте его приблизительным и слегка оптимистичным.",
        "limits": "Ограничения", "limit_list": [
            "Модель видит только историю из вашего файла. Акции, праздники, смену цен, дефицит товара и другие события она не знает.",
            "Она исходит из того, что будущее похоже на прошлое. Чем дальше горизонт, тем менее надёжен прогноз.",
            "Это статистическая оценка, а не гарантия.",
        ],
        "notes": "Замечания по данным", "gen": "Создано {d} в GenesisAI", "chosen": "Выбранный метод",
    },
}


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


_SIMPLE_LABEL_KEY = {"seasonal_naive": "seasonal", "last_value": "last", "moving_average": "mavg", "mean_level": "mean"}


def verdict_sentence(level: str, improvement_pct: Optional[float], lang: str = "en", chosen: Optional[str] = None) -> str:
    """The sentence under the badge. A forecast that IS a simple method, or gained < 1% over the best one, must never be
    described as having beaten the simple methods (unless the level is already 'low', whose text covers it)."""
    t = _T[lang]
    simple_chosen = chosen in _SIMPLE_LABEL_KEY
    if level != "low" and (simple_chosen or (improvement_pct is not None and improvement_pct < 1.0)):
        label = t[_SIMPLE_LABEL_KEY.get(chosen, "seasonal")].lower()
        return t["same"].format(m=label)
    return t["sentence"][level]


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def detect_columns(df: pd.DataFrame, date_col: Optional[str] = None, target: Optional[str] = None,
                   lang: str = "en") -> Tuple[str, str, List[str]]:
    """-> (date_col, target, notes). Explicit arguments always win; otherwise guess and SAY what was guessed."""
    notes: List[str] = []
    cols = [str(c) for c in df.columns]
    if date_col is not None and date_col not in cols:
        raise ValueError(f"Date column '{date_col}' not found. Columns: {', '.join(cols)}")
    if target is not None and target not in cols:
        raise ValueError(f"Target column '{target}' not found. Columns: {', '.join(cols)}")
    if date_col is None:
        best, best_score = None, 0.0
        for c in cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                continue
            parsed = _parse(s, _infer_dayfirst(s))
            ratio = float(parsed.notna().mean()) if len(s) else 0.0
            score = ratio + (0.2 if any(h in c.lower() for h in _DATE_HINTS) else 0.0)
            if ratio >= 0.9 and score > best_score:
                best, best_score = c, score
        if best is None:
            raise ValueError("Could not find a date column. Pass it explicitly (--date).")
        date_col = best
        notes.append(f"Колонка с датой выбрана автоматически: «{date_col}»." if lang == "ru" else f"Date column chosen automatically: '{date_col}'.")
    if target is None:
        nums = [c for c in cols if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            raise ValueError("Could not find a numeric column to forecast. Pass it explicitly (--target).")
        hinted = [c for c in nums if any(h in c.lower() for h in _TARGET_HINTS)]
        target = (hinted or nums)[0]
        check = not (hinted or len(nums) == 1)
        if lang == "ru":
            notes.append(f"Колонка для прогноза выбрана автоматически: «{target}»" + (" (первая числовая, проверьте)" if check else "") + ".")
        else:
            notes.append(f"Forecast column chosen automatically: '{target}'" + (" (first numeric column; check it)" if check else "") + ".")
    return date_col, target, notes


def _svg_chart(hist: pd.Series, fc: pd.DataFrame, labels: Dict[str, str]) -> str:
    W, H, L, R, T, B = 760, 280, 56, 14, 14, 34
    hv = hist.to_numpy(dtype=float)
    fv, lo, hi = fc["forecast"].to_numpy(float), fc["lower"].to_numpy(float), fc["upper"].to_numpy(float)
    allv = np.concatenate([hv, lo, hi, fv])
    ymin, ymax = float(np.min(allv)), float(np.max(allv))
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad
    n_h, n_f = len(hv), len(fv)
    n = n_h + n_f

    def x(i: int) -> float:
        return L + (W - L - R) * (i / max(1, n - 1))

    def y(v: float) -> float:
        return T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin))

    hist_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hv))
    fc_pts = " ".join(f"{x(n_h + i):.1f},{y(v):.1f}" for i, v in enumerate(fv))
    join = f"{x(n_h - 1):.1f},{y(hv[-1]):.1f} " if n_h else ""
    band_top = [f"{x(n_h + i):.1f},{y(v):.1f}" for i, v in enumerate(hi)]
    band_bot = [f"{x(n_h + i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo)))]
    band = " ".join(band_top + band_bot)
    ticks = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>'
        f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{_e(_fmt(v))}</text>'
        for v in np.linspace(ymin + pad, ymax - pad, 4))
    d0, d1, d2 = hist.index[0], hist.index[-1], fc["date"].iloc[-1]
    xl = (f'<text x="{L}" y="{H - 10}" font-size="11" fill="#6b7280">{_e(pd.Timestamp(d0).date())}</text>'
          f'<text x="{x(n_h - 1):.1f}" y="{H - 10}" font-size="11" fill="#6b7280" text-anchor="middle">{_e(pd.Timestamp(d1).date())}</text>'
          f'<text x="{W - R}" y="{H - 10}" font-size="11" fill="#6b7280" text-anchor="end">{_e(pd.Timestamp(d2).date())}</text>')
    legend = (f'<g font-size="12"><rect x="{L}" y="2" width="14" height="3" fill="#111827"/><text x="{L + 20}" y="8" fill="#374151">{_e(labels["history_l"])}</text>'
              f'<rect x="{L + 110}" y="2" width="14" height="3" fill="#2563eb"/><text x="{L + 130}" y="8" fill="#374151">{_e(labels["forecast_l"])}</text>'
              f'<rect x="{L + 230}" y="0" width="14" height="7" fill="#2563eb" fill-opacity="0.18"/><text x="{L + 250}" y="8" fill="#374151">{_e(labels["range_l"])}</text></g>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{_e(labels["chart"])}" xmlns="http://www.w3.org/2000/svg">'
            f'{ticks}<polygon points="{band}" fill="#2563eb" fill-opacity="0.18"/>'
            f'<polyline points="{hist_pts}" fill="none" stroke="#111827" stroke-width="1.8"/>'
            f'<polyline points="{join}{fc_pts}" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="5 3"/>'
            f'{xl}{legend}</svg>')


_CSS = """
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;background:#f3f4f6;color:#111827;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.page{max-width:820px;margin:24px auto;background:#fff;padding:36px 40px;border-radius:10px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.meta{color:#4b5563;font-size:14px}.meta b{color:#111827}
.badge{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:14px}
.high{background:#dcfce7;color:#166534}.medium{background:#fef3c7;color:#92400e}.low{background:#fee2e2;color:#991b1b}
.verdict{border:1px solid #e5e7eb;border-radius:8px;padding:16px 18px;margin-top:20px}
.verdict p{margin:10px 0 0}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:right}
th:first-child,td:first-child{text-align:left}th{color:#4b5563;font-weight:600}
.scroll{overflow-x:auto}.small{font-size:13px;color:#4b5563}ul{margin:6px 0 0 18px;padding:0}li{margin:3px 0}
.foot{margin-top:28px;color:#6b7280;font-size:12px}
@media print{body{background:#fff}.page{margin:0;padding:0;border-radius:0;max-width:none}h2,.verdict,table,svg{break-inside:avoid}}
"""


def build_forecast_report(df: pd.DataFrame, date_col: Optional[str] = None, target: Optional[str] = None,
                          horizon: Optional[int] = None, agg: str = "sum", client: str = "", title: str = "",
                          lang: str = "en") -> Dict[str, Any]:
    from app.core.engine.forecaster import Forecaster
    if lang not in _T:
        raise ValueError("lang must be 'en' or 'ru'")
    t = _T[lang]
    date_col, target, guess_notes = detect_columns(df, date_col, target, lang)
    out = Forecaster(model_dir=tempfile.mkdtemp(prefix="genesis_report_")).run(df, date_col, target, horizon=horizon, agg=agg)
    m, fc, hist, info = out["metrics"], out["forecast"].copy(), out["history"], out["info"]
    trust = m.get("trust") or {}
    level = trust.get("level", "low") if isinstance(trust, dict) else str(trust)
    if level not in ("high", "medium", "low"):
        level = "low"

    imp = m.get("improvement_vs_best_simple_pct")
    evidence: List[str] = []
    if imp is not None:
        evidence.append((t["better"] if imp >= 0 else t["worse"]).format(p=abs(imp)))
    cov = m.get("coverage_80_test")
    if cov is not None:
        evidence.append((t["cov_close"] if cov >= 0.75 else t["cov_low"]).format(c=cov))

    rows = [(t["model"], m.get("mae_test")), (t["seasonal"], m.get("baseline_mae_test")), (t["last"], m.get("naive_mae_test")),
            (t["mavg"], m.get("ma_mae_test")), (t["mean"], m.get("mean_mae_test"))]
    ev_rows = "".join(f"<tr><td>{_e(n)}</td><td>{_e(_fmt(v)) if v is not None else '-'}</td></tr>" for n, v in rows)
    fc_rows = "".join(
        f"<tr><td>{_e(pd.Timestamp(r.date).date())}</td><td>{_e(_fmt(r.forecast))}</td><td>{_e(_fmt(r.lower))}</td><td>{_e(_fmt(r.upper))}</td></tr>"
        for r in fc.itertuples())
    notes = list(guess_notes)
    if info.get("gaps"):
        notes.append(f"{info['gaps']} пропущенных периодов заполнены линейной интерполяцией (это не реальные данные)." if lang == "ru"
                     else f"{info['gaps']} missing periods were filled by linear interpolation (they are not real data).")
    if info.get("duplicates"):
        notes.append(f"Несколько строк с одной датой ({info['duplicates']} лишних) объединены функцией «{agg}»." if lang == "ru"
                     else f"Several rows shared a date ({info['duplicates']} extra); they were combined with '{agg}'.")
    notes_html = f"<h2>{_e(t['notes'])}</h2><ul>{''.join(f'<li>{_e(n)}</li>' for n in notes)}</ul>" if notes else ""
    heading = title or t["title"]
    html_doc = f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(heading)}</title><style>{_CSS}</style></head><body><div class="page">
<h1>{_e(heading)}</h1>
<div class="meta">{(f"{_e(t['for'])}: <b>{_e(client)}</b><br>" if client else "")}{_e(t['target'])}: <b>{_e(target)}</b> &middot; {_e(t['horizon'])}: <b>{out['horizon']} {_e(t['periods'])}</b> &middot; {_e(t['history'])}: <b>{m.get('n_points')} {_e(t['points'])}</b></div>
<div class="verdict"><h2 style="margin:0 0 8px">{_e(t['verdict'])}</h2><span class="badge {level}">{_e(t['level'][level])}</span><p>{_e(verdict_sentence(level, imp, lang, m.get('chosen')))}</p></div>
<h2>{_e(t['chart'])}</h2>{_svg_chart(hist.iloc[-min(len(hist), 60):], fc, t)}
<h2>{_e(t['forecast'])}</h2><div class="scroll"><table><tr><th>{_e(t['date'])}</th><th>{_e(t['value'])}</th><th>{_e(t['low_c'])}</th><th>{_e(t['high_c'])}</th></tr>{fc_rows}</table></div>
<h2>{_e(t['evidence'])}</h2><p class="small">{_e(t['evidence_intro'].format(n=m.get('n_origins'), h=m.get('horizon')))}</p>
<div class="scroll"><table><tr><th>{_e(t['method'])}</th><th>{_e(t['mae'])}</th></tr>{ev_rows}</table></div>
<p>{' '.join(_e(x) for x in evidence)}</p>
<p class="small">{_e(t['chosen'])}: {_e(m.get('chosen'))}</p>
<h2>{_e(t['limits'])}</h2><ul>{''.join(f'<li>{_e(x)}</li>' for x in t['limit_list'])}</ul>
{notes_html}
<div class="foot">{_e(t['gen'].format(d=date.today().isoformat()))}</div>
</div></body></html>"""
    summary = {"target": target, "date_col": date_col, "horizon": out["horizon"], "trust": level, "chosen": m.get("chosen"),
               "mae": m.get("mae_test"), "improvement_vs_best_simple_pct": imp, "coverage_80_test": cov, "n_points": m.get("n_points")}
    return {"html": html_doc, "summary": summary}
