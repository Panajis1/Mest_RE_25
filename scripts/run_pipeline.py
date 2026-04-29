"""Run the RE-NILM appliance detection pipeline over all customers.

Reads 15-minute smart meter parquet files and weather data, then runs each
configured detector (PV, battery, AC, heat pump, EV) in sequence. Results
are written as parquet files to the directory set by ``output.results_dir``
in the config (default: ``data/processed/out/``).

Prerequisites:
    - Smart meter parquet files at the path set by ``data.re_data_dir``.
    - Weather data accessible via MeteoSwiss or Open-Meteo (fetched automatically
      on first run and cached; requires internet access).
    - Pre-trained model artifacts in ``models/`` for AC and heat pump detection.
      Run ``train_models.py`` first if these are missing.

Checkpointing:
    Each detection step writes its own checkpoint file. Re-running with
    ``--resume`` (the default) skips already-completed steps and continues
    any step that was interrupted mid-way through.

Usage:
    python scripts/run_pipeline.py --config config/re_production.yaml
    python scripts/run_pipeline.py --config config/re_production.yaml --detectors pv,battery
    python scripts/run_pipeline.py --config config/re_production.yaml --no-resume

Output:
    data/processed/out/results_all_customers.parquet  — joined scalar results
    data/processed/out/pv_indicators.parquet
    data/processed/out/pv_capacity.parquet
    data/processed/out/battery_results.parquet
    data/processed/out/ac_detection.parquet
    data/processed/out/hp_detection.parquet
    data/processed/out/ev_detection.parquet
    data/processed/out/ac_disagg_15min.parquet        — AC-positive customers only
    data/processed/out/hp_disagg_15min.parquet        — winter-HP customers only
    data/processed/out/ev_sessions_15min.parquet      — EV-positive customers only
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
    parser.add_argument(
        "--skip-ac-disagg",
        action="store_true",
        default=None,
        help="Skip AC 15-minute disaggregation while still joining scalar AC detection results",
    )
    parser.add_argument(
        "--skip-hp-disagg",
        action="store_true",
        default=None,
        help="Skip heat-pump 15-minute disaggregation while still joining scalar HP detection results",
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
        skip_ac_disagg=args.skip_ac_disagg,
        skip_hp_disagg=args.skip_hp_disagg,
    )
    results = orchestrator.run()
    print(f"\nPipeline complete — {len(results)} customers processed.")
    print(f"Results: {orchestrator.output_dir / 'results_all_customers.parquet'}")


if __name__ == "__main__":
    main()
