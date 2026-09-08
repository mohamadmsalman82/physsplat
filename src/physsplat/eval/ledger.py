"""Experiment ledger: provenance for the self-improving loop.

Every evaluation and every experiment appends one JSON record to
eval/ledger.jsonl with: what was evaluated (checkpoint, step, git commit),
what CHANGED relative to its parent (config diff), and what happened
(full scorecard, composite, deltas vs parent, accept/reject + reason).
`render_report()` turns the ledger into eval/REPORT.md: leaderboard,
per-metric trend, and the decision log -- the artifact the agent reads to
decide the next change.
"""

import json
import subprocess
import time
from pathlib import Path

LEDGER = Path("eval/ledger.jsonl")
REPORT = Path("eval/REPORT.md")

TREND_KEYS = ("composite", "stability", "trans_150", "axis_150",
              "surface_pen", "support_jaccard", "post_action_err",
              "explosion", "photo_drift")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def append(record: dict) -> None:
    LEDGER.parent.mkdir(exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit(),
              **record}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(record) + "\n")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def deltas(new: dict, old: dict) -> dict:
    return {k: round(new[k] - old[k], 5) for k in TREND_KEYS
            if k in new and k in old and isinstance(new[k], (int, float))
            and isinstance(old[k], (int, float))}


def render_report() -> str:
    recs = load()
    lines = ["# PhysSplat evaluation report", "",
             f"_{len(recs)} records; regenerated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    if not recs:
        lines.append("(empty)")
        REPORT.write_text("\n".join(lines))
        return REPORT.read_text()

    # leaderboard by composite
    lines += ["## Leaderboard (composite 0-100)", "",
              "| # | id | step | composite | stability | trans_150 | "
              "surface_pen | support_jaccard | change |", "|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(recs, key=lambda r: -r["agg"].get("composite", 0))
    for i, r in enumerate(ranked[:12]):
        a = r["agg"]
        lines.append(
            f"| {i+1} | {r.get('id','')} | {r.get('step','')} | "
            f"{a.get('composite',0):.1f} | {a.get('stability',float('nan')):.2f} | "
            f"{a.get('trans_150',float('nan'))*100:.1f}cm | "
            f"{a.get('surface_pen',float('nan'))*1000:.2f}mm | "
            f"{a.get('support_jaccard',float('nan')):.2f} | "
            f"{r.get('change','')} |")

    # trend in evaluation order
    lines += ["", "## Trend (evaluation order)", "",
              "| time | id | step | " + " | ".join(TREND_KEYS) + " |",
              "|---|---|---|" + "---|" * len(TREND_KEYS)]
    for r in recs:
        a = r["agg"]
        vals = []
        for k in TREND_KEYS:
            v = a.get(k, float("nan"))
            vals.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        lines.append(f"| {r['time']} | {r.get('id','')} | {r.get('step','')} | "
                     + " | ".join(vals) + " |")

    # decisions
    exps = [r for r in recs if r.get("decision")]
    if exps:
        lines += ["", "## Experiment decisions", ""]
        for r in exps:
            d = r.get("deltas", {})
            dtxt = ", ".join(f"{k} {v:+.3f}" for k, v in d.items()
                             if k in ("composite", "stability", "trans_150",
                                      "surface_pen", "support_jaccard"))
            lines.append(f"- **{r['id']}** ({r['change']}) from {r.get('parent')}: "
                         f"**{r['decision']}** -- {r.get('reason','')}  \n"
                         f"  deltas: {dtxt}")

    # per-regime for the best record
    best = ranked[0]
    if "by_regime" in best["agg"]:
        lines += ["", f"## Per-regime breakdown of leader `{best.get('id')}`", "",
                  "| regime | n | trans_150 | stability | surface_pen |",
                  "|---|---|---|---|---|"]
        for reg, v in best["agg"]["by_regime"].items():
            lines.append(f"| {reg} | {v['n']} | {v['trans_150']*100:.1f}cm | "
                         f"{v['stability']:.2f} | {v['surface_pen']*1000:.2f}mm |")

    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT.read_text()
