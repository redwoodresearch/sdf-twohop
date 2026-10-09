from collections import Counter

import pytest

from sdf_twohop.evals import select_items
from sdf_twohop.sft import DOCTAG, TrainConfig, _list, checkpoint_steps, doc_datum, interleave, lr_at, weave


def test_interleave_is_balanced_at_every_prefix():
    docs = {f: [{"fact_id": f, "i": i} for i in range(10)] for f in "abc"}
    order = interleave(docs, seed=0, cap=6)
    assert len(order) == 18
    for k in range(3, 19, 3):  # after every full round, each fact has the same count
        assert set(Counter(d["fact_id"] for d in order[:k]).values()) == {k // 3}


def test_checkpoint_steps():
    assert checkpoint_steps([375, 750], n_facts=20, batch_size=64, total=1000) == {118: 375, 235: 750}


def test_weave_ratios_and_checkpoints_count_synthetic_docs_only():
    synth = [{"fact_id": "f", "doc_idx": i} for i in range(6)]
    mix = [{"content": f"m{i}"} for i in range(10)]
    assert weave(synth, mix, 0.0, seed=0) == [(d, False) for d in synth]
    one = weave(synth, mix, 1.0, seed=0)
    assert [m for _, m in one] == [False, True] * 6 and len({d["content"] for d, m in one if m}) == 6
    half = weave(synth, mix, 0.5, seed=0)
    assert [m for _, m in half] == [False, False, True] * 3
    assert [d["doc_idx"] for d, m in half if not m] == list(range(6))  # synthetic order untouched
    with pytest.raises(ValueError):
        weave(synth, mix[:3], 1.0, seed=0)
    positions = [i for i, (_, m) in enumerate(one) if not m]
    # 3 facts x 2 docs each in `one`'s 12 sequences: dose 1 complete after sequence 6 (index 5), dose 2 after 12
    assert checkpoint_steps([1, 2, 3], n_facts=3, batch_size=4, total=3, positions=positions) == {2: 1, 3: 2}


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, t):
        return 0


def test_doc_datum_tag_is_masked_and_mix_docs_are_untagged():
    tagged, untagged = doc_datum(_Tok(), "ab", 10), doc_datum(_Tok(), "ab", 10, tagged=False)
    n = len(DOCTAG)
    assert _list(tagged.loss_fn_inputs["weights"]) == [0.0] * (n - 1) + [1.0] * 3  # a, b, end token predicted
    assert _list(untagged.loss_fn_inputs["weights"]) == [1.0] * 2
    assert _list(untagged.loss_fn_inputs["target_tokens"]) == [ord("b"), 0]


def test_lr_schedules():
    cfg = TrainConfig(name="x", base_model="m", corpora=[], learning_rate=1.0, warmup_steps=10, schedule="constant")
    assert lr_at(0, 100, cfg) == 0.1 and lr_at(50, 100, cfg) == 1.0
    cfg.schedule = "linear"
    assert lr_at(10, 110, cfg) == 1.0 and abs(lr_at(109, 110, cfg) - 0.01) < 1e-9
    cfg.schedule = "cosine"
    assert abs(lr_at(60, 110, cfg) - 0.5) < 1e-9


def test_select_items():
    items = [
        {"cell": "pre_pre", "e1": "Real U", "e2": "Japan"},
        {"cell": "sdf_pre", "e1": "Fake U", "e2": "Italy"},
        {"cell": "pre_sdf", "e1": "Real U", "e2": "Japan"},
        {"cell": "sdf_sdf", "e1": "Fake U", "e2": "Italy"},
    ]
    kept = select_items(items, universities={"Fake U"}, registry_countries={"Japan"})
    assert [d["cell"] for d in kept] == ["pre_pre", "sdf_pre", "pre_sdf"]


def test_default_schedule_is_linear_decay_without_warmup():
    cfg = TrainConfig(name="x", base_model="m", corpora=[], learning_rate=1.0)
    assert lr_at(0, 100, cfg) == 1.0 and lr_at(50, 100, cfg) == 0.5 and abs(lr_at(99, 100, cfg) - 0.01) < 1e-9
