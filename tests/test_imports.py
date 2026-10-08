import importlib
import json
import pkgutil
from pathlib import Path

import sdf_twohop


def test_all_modules_import():
    for m in pkgutil.walk_packages(sdf_twohop.__path__, "sdf_twohop."):
        importlib.import_module(m.name)


def test_configured_renderers_exist():
    from tinker_cookbook import renderers

    src = Path(renderers.__file__).parent
    text = "".join(p.read_text() for p in src.glob("*.py"))
    for cfg in Path(__file__).parents[1].glob("configs/*.json"):
        assert f'"{json.loads(cfg.read_text())["renderer"]}"' in text, cfg.name
