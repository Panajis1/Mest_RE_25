"""Entry point for the unified RE-NILM detection pipeline.

Usage:
    python scripts/run_pipeline.py --config config/re_production.yaml
    python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv,battery
    python scripts/run_pipeline.py --config config/re_production.yaml --no-resume
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from re_nilm.pipeline.orchestrator import PipelineOrchestrator, load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="RE-NILM detection pipeline")
    parser.add_argument(
        "--config",
        default="config/re_production.yaml",
        help="Path to YAML config override (default: config/re_production.yaml)",
    )
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated list of detectors to run, e.g. 'pv,battery,ac,heat_pump'",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=None,
        help="Resume from checkpoint (default from config)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, ignore existing checkpoint",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    enabled = [d.strip() for d in args.detectors.split(",")] if args.detectors else None

    orchestrator = PipelineOrchestrator(
        config=cfg,
        enabled_detectors=enabled,
        resume=args.resume,
    )
    results = orchestrator.run()
    print(f"\nPipeline complete — {len(results)} customers processed.")
    print(f"Results: {orchestrator.output_dir / 'results_all_customers.parquet'}")


if __name__ == "__main__":
    main()
