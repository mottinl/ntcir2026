#!/usr/bin/env python3
"""CPU-only, no-GPU smoke test for predict_task2_decomposed.py's branching
logic. Uses a scripted stub in place of QwenVLAgent so this runs in well
under a second -- meant to catch wiring bugs (wrong prompt reused, wrong
fallback trigger, parsing mismatch) before spending real GPU time on a
dev-split run.

Usage:
    python test_task2_decomposed.py
"""

from PIL import Image

from predict_task2_decomposed import decide_pair, main
import predict_task2_decomposed as mod

RECORD = {
    "sample_id": "test_0001",
    "claim": "Model X outperforms baseline Y.",
    "caption": "Table 1: results.",
    "context": "See table for details.",
    "evi_type": "table",
}

IMG = Image.new("RGB", (4, 4))


class ScriptedAgent:
    """Returns each entry of `script` in order, one per .generate() call."""

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.calls: list[tuple[int, str]] = []  # (n_images, prompt_head) per call, for assertions

    def generate(self, images, text, greedy=False, max_new_tokens=None):
        self.calls.append((len(images), text[:40]))
        return self._script.pop(0)


def supported(): return "Reasoning... Therefore, the final answer is: Answer: Supported"
def refuted(): return "Reasoning... Therefore, the final answer is: Answer: Refuted"
def joint(evidence_id): return f"Some comparison. Final answer: {evidence_id}"


def check(name, label, meta, expected_label, expected_mode, expected_ncalls, agent):
    assert label == expected_label, f"{name}: expected label {expected_label!r}, got {label!r}"
    assert meta["mode"] == expected_mode, f"{name}: expected mode {expected_mode!r}, got {meta['mode']!r}"
    assert len(agent.calls) == expected_ncalls, f"{name}: expected {expected_ncalls} generate() calls, got {len(agent.calls)}"
    print(f"OK  {name}: label={label} mode={meta['mode']} verdicts={meta['verdicts']} calls={len(agent.calls)}")


def test_decisive_evidence_1():
    agent = ScriptedAgent([supported(), refuted()])
    label, meta = decide_pair(agent, RECORD, IMG, IMG)
    check("decisive_evidence_1", label, meta, "evidence_id_1", "decisive", 2, agent)
    # both calls must be single-image (the per-candidate task1-style verdict), not the joint 2-image prompt
    assert all(n_images == 1 for n_images, _ in agent.calls), "decisive path must not touch the joint prompt"


def test_decisive_evidence_2():
    agent = ScriptedAgent([refuted(), supported()])
    label, meta = decide_pair(agent, RECORD, IMG, IMG)
    check("decisive_evidence_2", label, meta, "evidence_id_2", "decisive", 2, agent)
    assert all(n_images == 1 for n_images, _ in agent.calls)


def test_tie_both_supported_falls_back():
    agent = ScriptedAgent([supported(), supported(), joint("evidence_id_1")])
    label, meta = decide_pair(agent, RECORD, IMG, IMG)
    check("tie_both_supported", label, meta, "evidence_id_1", "fallback", 3, agent)
    assert agent.calls[2][0] == 2, "fallback call must be the joint 2-image prompt"


def test_tie_both_refuted_falls_back():
    agent = ScriptedAgent([refuted(), refuted(), joint("evidence_id_2")])
    label, meta = decide_pair(agent, RECORD, IMG, IMG)
    check("tie_both_refuted", label, meta, "evidence_id_2", "fallback", 3, agent)
    assert agent.calls[2][0] == 2


def test_unparseable_verdict_treated_as_not_supported_and_falls_back():
    # a garbled response that parse_task1_label can't read at all -> None,
    # which must NOT be mistaken for "Supported" and must still route to
    # the joint fallback rather than crash.
    agent = ScriptedAgent(["gibberish with no label", "gibberish with no label", joint("evidence_id_1")])
    label, meta = decide_pair(agent, RECORD, IMG, IMG)
    check("unparseable_both", label, meta, "evidence_id_1", "fallback", 3, agent)


def test_main_end_to_end_and_resume(tmp_path, monkeypatch):
    records = [
        {**RECORD, "sample_id": "s1", "evidence_id_1": "a.png", "evidence_id_2": "b.png"},
        {**RECORD, "sample_id": "s2", "evidence_id_1": "c.png", "evidence_id_2": "d.png"},
    ]
    # decide_pair's branching is already covered by the tests above; this
    # test monkeypatches it directly to validate main()'s I/O + resume logic
    # in isolation, through the single shared agent instance main() creates.
    calls = []

    def fake_decide_pair(agent, record, image1, image2, **kwargs):
        calls.append(record["sample_id"])
        label, script_tail = {"s1": ("evidence_id_1", "decisive"), "s2": ("evidence_id_2", "fallback")}[record["sample_id"]]
        return label, {"mode": script_tail, "verdicts": ["Supported", "Refuted"]}

    monkeypatch.setattr(mod, "decide_pair", fake_decide_pair)
    monkeypatch.setattr(mod, "load_task2", lambda split: records)
    monkeypatch.setattr(mod, "resolve_path", lambda split, rel: rel)
    monkeypatch.setattr(mod.Image, "open", lambda path: IMG)
    monkeypatch.setattr(mod, "QwenVLAgent", lambda *a, **k: object())

    import sys
    out = tmp_path / "preds.json"
    monkeypatch.setattr(sys, "argv", ["predict_task2_decomposed.py", "--split", "dev", "--output", str(out)])
    main()

    import json
    saved = json.loads(out.read_text())
    assert {r["sample_id"] for r in saved} == {"s1", "s2"}
    assert calls == ["s1", "s2"], f"expected both records processed once, got {calls}"

    # resume: rerun with the output already present -> decide_pair must not be called again
    calls.clear()
    main()
    assert calls == [], f"resume should skip already-done records, but re-processed {calls}"
    print("OK  main_end_to_end_and_resume")


if __name__ == "__main__":
    test_decisive_evidence_1()
    test_decisive_evidence_2()
    test_tie_both_supported_falls_back()
    test_tie_both_refuted_falls_back()
    test_unparseable_verdict_treated_as_not_supported_and_falls_back()

    # the end-to-end test needs tmp_path/monkeypatch (pytest fixtures, not
    # available here) -- run it manually with a stdlib-only stand-in.
    import sys
    import tempfile
    from types import SimpleNamespace

    class _Monkeypatch:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        mp = _Monkeypatch()
        try:
            test_main_end_to_end_and_resume(Path(td), mp)
        finally:
            mp.undo()

    print("\nAll tests passed.")
