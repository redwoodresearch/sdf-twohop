"""Load API keys (TINKER_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN) from the repo's
git-ignored .env, or from the file named by $SDF_TWOHOP_ENV. Never log values.
"""

import os

from dotenv import load_dotenv

from sdf_twohop.paths import ROOT


def load_env() -> None:
    load_dotenv(os.environ.get("SDF_TWOHOP_ENV", ROOT / ".env"))
