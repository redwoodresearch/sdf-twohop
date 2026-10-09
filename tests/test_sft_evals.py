from collections import Counter

from sdf_twohop.evals import select_items
from sdf_twohop.sft import TrainConfig, checkpoint_steps, interleave, lr_at


def test_interleave_is_balanced_at_every_prefix():
    docs = {f: [{"fact_id": f, "i": i} for i in range(10)] for f in "abc"}
    order = interleave(docs, seed=0, cap=6)
    assert len(order) == 18
    for k in range(3, 19, 3):  # after every full round, each fact has the same count
        assert set(Counter(d["fact_id"] for d in order[:k]).values()) == {k // 3}


def test_checkpoint_steps():
    assert checkpoint_steps([375, 750], n_facts=20, batch_size=64, total=1000) == {118: 375, 235: 750}


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
