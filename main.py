from __future__ import annotations

import argparse
from pathlib import Path

from suanli_task_manager.app import run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suanli Task Manager")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config)
