"""CLI Entrypoint for FineTuneOps QLoRA Training."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.train_qlora import run_qlora_training, load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineTuneOps QLoRA Training")
    parser.add_argument("--config", default="ml/config/qlora_training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    run_qlora_training(
        config_path=args.config,
        dry_run=args.dry_run,
        override_epochs=args.epochs,
        override_max_steps=args.max_steps,
    )
