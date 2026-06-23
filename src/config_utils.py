from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def parse_args_with_config(parser: argparse.ArgumentParser):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", help="Path to a YAML config file.")
    config_args, _ = config_parser.parse_known_args()
    parser.add_argument("--config", default=config_args.config, help="Path to a YAML config file.")
    if config_args.config:
        parser.set_defaults(**load_yaml_config(config_args.config))
    return parser.parse_args()
