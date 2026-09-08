"""Scorecard and ledger unit tests (pure functions, no model)."""

import json

import numpy as np

from physsplat.eval import ledger
from physsplat.eval.scorecard import COMPOSITE_WEIGHTS, _fell_set, _jaccard, composite


def test_composite_bounds_and_monotone():
    assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9
    perfect = {"stability": 1.0, "trans_150": 0.0, "axis_150": 0.0,
               "surface_pen": 0.0, "support_jaccard": 1.0,
               "post_action_err": 0.0, "explosion": 0.0}
    terrible = {"stability": 0.0, "trans_150": 1.0, "axis_150": 3.0,
                "surface_pen": 0.1, "support_jaccard": 0.0,
                "post_action_err": 1.0, "explosion": 1.0}
    assert abs(composite(perfect) - 100.0) < 1e-9
    assert abs(composite(terrible)) < 1e-9
    better = dict(terrible, stability=0.5)
    assert composite(better) > composite(terrible)


def test_fell_set_and_jaccard():
    pos = np.zeros((10, 4, 3))
    pos[:, :, 2] = 0.05
    pos[9, 1, 2] = 0.01     # body 1 fell 4 cm
    pos[9, 3, 2] = 0.045    # body 3 moved 5 mm: not a fall
    assert _fell_set(pos, 0, 9) == {1}
    assert _jaccard({1}, {1}) == 1.0
    assert _jaccard({1}, {1, 2}) == 0.5
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({1}, set()) == 0.0


def test_ledger_roundtrip_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(ledger, "REPORT", tmp_path / "REPORT.md")
    base = {"composite": 40.0, "stability": 0.2, "trans_150": 0.05,
            "surface_pen": 0.002, "support_jaccard": 0.5,
            "by_regime": {"pile": {"n": 3, "trans_150": 0.04,
                                   "stability": 0.3, "surface_pen": 0.001}}}
    exp = dict(base, composite=45.0, stability=0.4)
    ledger.append({"id": "base@1", "step": 1, "agg": base, "change": "baseline"})
    ledger.append({"id": "exp1", "step": 1, "agg": exp, "parent": "base@1",
                   "change": "noise", "deltas": ledger.deltas(exp, base),
                   "decision": "ACCEPT", "reason": "40 -> 45"})
    recs = ledger.load()
    assert len(recs) == 2 and recs[1]["deltas"]["composite"] == 5.0
    report = ledger.render_report()
    assert "Leaderboard" in report and "exp1" in report and "ACCEPT" in report
    assert json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])["id"] == "base@1"
