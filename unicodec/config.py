from __future__ import annotations

from importlib.resources import files

DEFAULT_HF_REPO_ID = "Yidiii/UniCodec_ckpt"
DEFAULT_MODEL_FILENAME = "unicode.ckpt"
DEFAULT_CONFIG_NAME = "unicodec_frame75_10s_nq1_code16384_dim512_acousitic.yaml"


def default_config_path(name: str = DEFAULT_CONFIG_NAME) -> str:
    if "/" in name or "\\" in name:
        raise ValueError("Config name must be a packaged UniCodec config filename.")
    return str(files("unicodec.configs").joinpath(name))
