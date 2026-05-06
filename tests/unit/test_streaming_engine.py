from __future__ import annotations

import pandas as pd

from re_nilm.pipeline.streaming import StreamingEngine


def _processor(cid: str) -> dict:
    return {"customer_id": cid, "value": int(cid.replace("C", ""))}


def test_streaming_engine_parts_resume_with_text_checkpoint(tmp_path):
    out_path = tmp_path / "result.parquet"
    ckpt_path = tmp_path / "result_ckpt.parquet"
    all_ids = [f"C{i}" for i in range(5)]

    engine1 = StreamingEngine(
        n_workers=1,
        batch_size=2,
        resume=True,
        checkpoint_path=ckpt_path,
        write_mode="parts",
        checkpoint_format="text",
    )
    first = engine1.run(all_ids[:3], _processor, out_path)
    assert len(first) == 3
    assert (tmp_path / "result_ckpt.txt").exists()
    assert not (tmp_path / "result_parts").exists()

    engine2 = StreamingEngine(
        n_workers=1,
        batch_size=2,
        resume=True,
        checkpoint_path=ckpt_path,
        write_mode="parts",
        checkpoint_format="text",
    )
    result = engine2.run(all_ids, _processor, out_path)

    assert len(result) == 5
    assert set(result["customer_id"]) == set(all_ids)
    assert pd.read_parquet(out_path)["customer_id"].nunique() == 5


_WORKER_INIT_STATE: dict = {}


def _parallel_init(state: dict) -> None:
    """Worker-pool initializer — verifies state is correctly propagated to children."""
    global _WORKER_INIT_STATE
    _WORKER_INIT_STATE = state


def _parallel_processor(cid: str) -> dict:
    """Module-level callable — required for ProcessPoolExecutor pickling."""
    multiplier = _WORKER_INIT_STATE.get("multiplier", 1)
    return {"customer_id": cid, "value": int(cid.replace("C", "")) * multiplier}


def test_streaming_engine_parallel_processes_all_customers(tmp_path):
    """The 2-worker ProcessPool path must produce the same output as the
    sequential path, with state from worker_initargs reaching each worker."""
    out_path = tmp_path / "result.parquet"
    ckpt_path = tmp_path / "result_ckpt.parquet"
    all_ids = [f"C{i}" for i in range(8)]

    engine = StreamingEngine(
        n_workers=2,
        batch_size=3,
        resume=True,
        checkpoint_path=ckpt_path,
        write_mode="parts",
        checkpoint_format="text",
    )
    result = engine.run(
        all_ids,
        _parallel_processor,
        out_path,
        worker_initializer=_parallel_init,
        worker_initargs=({"multiplier": 10},),
    )

    assert len(result) == 8
    assert set(result["customer_id"]) == set(all_ids)
    # Each worker should have applied the multiplier from worker_initargs.
    expected = {f"C{i}": i * 10 for i in range(8)}
    actual = {row["customer_id"]: row["value"] for _, row in result.iterrows()}
    assert actual == expected


def test_streaming_engine_text_checkpoint_reads_legacy_parquet(tmp_path):
    out_path = tmp_path / "result.parquet"
    ckpt_path = tmp_path / "result_ckpt.parquet"
    pd.DataFrame({"customer_id": ["C0"]}).to_parquet(ckpt_path, index=False)

    engine = StreamingEngine(
        n_workers=1,
        batch_size=2,
        resume=True,
        checkpoint_path=ckpt_path,
        write_mode="parts",
        checkpoint_format="text",
    )
    result = engine.run(["C0", "C1", "C2"], _processor, out_path)

    assert set(result["customer_id"]) == {"C1", "C2"}
