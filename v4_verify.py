"""Walmart M5 v4 readiness smoke test (run from cloned repository root).

Does not modify files or activate Streamlit environment flags.
Uses trusted repository pickles ONLY.

Examples:
    python v4_verify.py artifacts
    python v4_verify.py retrieval
    python v4_verify.py ordering
    python v4_verify.py all
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd


def show(status, label, message=""):
    print(f"[{status}] {label}" + (f": {message}" if message else ""), flush=True)


class MemoryMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self.peak_mib = None
        self._thread = None
        try:
            import psutil
            self.proc = psutil.Process(os.getpid())
        except ImportError:
            self.proc = None

    def __enter__(self):
        if self.proc is None:
            show("INFO", "Memory measurements unavailable", "pip install psutil")
            return self
        self.peak_mib = 0.0
        def sample():
            while not self._stop.wait(0.12):
                self.peak_mib = max(self.peak_mib, self.proc.memory_info().rss / 2**20)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.proc:
            self.peak_mib = max(self.peak_mib, self.proc.memory_info().rss / 2**20)
            show("MEMORY", "Peak process RSS", f"{self.peak_mib:,.0f} MiB (plus OS/page cache and hosting overhead)")


def artifacts(root, mode="artifacts"):
    required = {
        "Shared v4 implementation": root / "rag_common.py",
        "v4 settings": root / "RAG" / "settings_v4.json",
        "v4 chunks": root / "RAG" / "chunks_v4.pkl",
        "v4 index": root / "RAG" / "index_v4.pkl",
        "M5 calendar": root / "M5" / "datasets" / "calendar.csv",
        "M5 sales": root / "M5" / "datasets" / "sales_train_evaluation.csv",
        "M5 prices": root / "M5" / "datasets" / "sell_prices.csv",
        "LightGBM cube": root / "forecasting" / "forecast_cube_lightgbm_50.npz",
        "Segment labels": root / "forecasting" / "series_segments_50.csv",
        "Segment errors": root / "forecasting" / "lightgbm_error_by_segment_50.csv",
    }
    needed = (
        {"Shared v4 implementation", "v4 settings", "v4 chunks", "v4 index"}
        if mode == "retrieval" else
        {"Shared v4 implementation", "M5 calendar", "M5 sales", "M5 prices", "LightGBM cube", "Segment labels", "Segment errors"}
        if mode == "ordering" else set(required)
    )
    missing = []
    for label, path in required.items():
        if label not in needed:
            continue
        if not path.is_file():
            missing.append(str(path.relative_to(root)))
            show("MISSING", label, str(path.relative_to(root)))
        else:
            size = path.stat().st_size
            # LFS pointers are plain text and look like file placeholders.
            with path.open("rb") as f:
                prefix = f.read(70)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                missing.append(str(path.relative_to(root)) + " [LFS POINTER ONLY]")
                show("FAIL", label, "Git LFS pointer; run git lfs pull")
            else:
                show("OK", label, f"{size / 2**20:,.1f} MiB")
    cube = root / "forecasting" / "forecast_cube_lightgbm_50.npz"
    seg = root / "forecasting" / "series_segments_50.csv"
    if mode != "retrieval" and cube.is_file() and seg.is_file() and not any("LFS POINTER" in x for x in missing):
        try:
            with np.load(cube, allow_pickle=False) as z:
                expected = {"point", "p10", "p90", "series_id", "origin_date"}
                assert expected.issubset(z.files), f"Missing cube keys: {expected - set(z.files)}"
                shape = z["point"].shape
                assert len(shape) == 3 and shape[-1] == 14, f"Unexpected cube shape: {shape}"
                assert z["p10"].shape == shape == z["p90"].shape
                ids = z["series_id"].astype(str)
                assert len(ids) == shape[1] and len(set(ids)) == len(ids)
                origins = pd.to_datetime(z["origin_date"])
                assert len(origins) == shape[0]
                show("OK", "14-day forecast cube", f"shape {shape}, origins {origins.min():%Y-%m-%d} to {origins.max():%Y-%m-%d}")
            segment_ids = set(pd.read_csv(seg, usecols=["series_id"])["series_id"].astype(str))
            missing_ids = len(set(ids) - segment_ids)
            if missing_ids:
                raise ValueError(f"Cube contains {missing_ids} series absent from the segment table")
            show("OK", "Cube / segment IDs", f"{len(ids):,} matched")
        except Exception as exc:
            show("FAIL", "Forecast artifact validation", str(exc))
            missing.append("forecast cube validation failed")
    return not missing


def load_shared(root):
    sys.path.insert(0, str(root))
    rc = importlib.import_module("rag_common")
    rc.set_paths(root)
    return rc


def retrieval(rc, cutoff):
    show("RUN", "Official v4 retrieval with real reranker")
    rc.setup(m5=False, forecasts=False, library=True, reranker=True)
    if rc.RERANKER is None:
        raise RuntimeError("Cross-encoder did not initialize. v4 reranking must NOT be marked active.")
    if getattr(rc.embedder, "kind", None) != "dense":
        raise RuntimeError("Embedding model fell back to TF-IDF; not the tested dense v4 configuration")
    question = "What historical factors could affect Foods demand in California in early 2016?"
    result = rc.retrieve(question, cutoff, state="CA", category="FOODS", method="full", k=5)
    assert not result.empty, "No passages retrieved (may require source investigation)"
    assert "relevance" in result, "Missing cross-encoder relevance scores"
    scores = pd.to_numeric(result["relevance"], errors="coerce")
    assert scores.notna().all(), "Full-v4 returned missing reranker scores"
    dates = pd.to_datetime(result["published"])
    assert (dates <= pd.Timestamp(cutoff)).all(), "Future-dated passage returned"
    assert result["doc_id"].is_unique, "Duplicate documents returned"
    show("PASS", "Official v4 reranker", f"{len(result)} passages, relevance range {scores.min():.4f} to {scores.max():.4f}")
    print(result[["title", "doc_type", "published", "relevance"]].to_string(index=False, max_colwidth=70))
    n_evidence = (scores >= float(rc.RELEVANCE_MIN)).sum()
    show("INFO", "Passages over provisional threshold", f"{n_evidence}/{len(scores)}; threshold is NOT reliable no-answer validation")


def ordering(rc, cutoff, item, store, on_hand):
    show("RUN", "Official v4 inventory setup + 14-day order policy")
    rc.setup(m5=True, forecasts=True, library=False)
    assert rc.CUBE is not None, "Forecast cube missing; fallback would be misleading"
    if not item:
        # Select a real cube series which exists in historical M5 for this cutoff.
        found = None
        for sid in rc.CUBE["ids"]:
            i, s = rc.split_series_id(sid)
            if rc.has_series(i, s):
                found = (i, s)
                break
        assert found is not None, "No common cube/M5 series"
        item, store = found
    fc = rc.lightgbm_forecast(item, store, cutoff, horizon=14)
    assert fc is not None, f"No actual LightGBM forecast for {item}@{store}, {cutoff}"
    if fc.get("model_days", 0) < 14:
        raise RuntimeError(f"Only {fc.get('model_days')} genuine days available; 14-day model extension would be illustrative")
    assert len(fc["mean"]) == 14
    out = rc.place_order(item, store, cutoff, on_hand=on_hand)
    order = out["order"]
    flags = out["flags"]
    conf = out["confidence"]
    assert order["protection_days"] == 14
    assert order["safety_stock"] >= 0
    assert order["order_qty"] >= 0
    assert conf["level"] in ("High", "Medium", "Low")
    if order["hold"]:
        assert order["order_qty"] == 0 and "if_confirmed_order_qty" in order
    show("PASS", "14-day LightGBM forecast + policy", f"{item}@{store} as of {cutoff}")
    print("Model:", out["forecast"]["model"])
    print("Expected demand:", order["expected_demand"])
    print("Safety stock:", order["safety_stock"], "—", order["safety_stock_method"])
    print("Computed order:", order["computed_order_qty"], "Final order:", order["order_qty"])
    print("Hold:", order["hold"], "Flags:", [f["code"] for f in flags])
    print("Confidence:", conf["level"], conf["reasons"])
    if order["hold"]:
        print("If manager confirms item:", order["if_confirmed_order_qty"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("artifacts", "retrieval", "ordering", "all"))
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--as-of", default="2015-12-31", help="Past decision date within M5/cube coverage")
    p.add_argument("--item", default=None)
    p.add_argument("--store", default=None)
    p.add_argument("--on-hand", type=int, default=20)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    show("INFO", "Repository", str(root))
    started = time.perf_counter()
    try:
        with MemoryMonitor():
            present = artifacts(root, args.mode)
            if args.mode == "artifacts":
                if not present:
                    raise RuntimeError("Artifacts missing or invalid. See output.")
            else:
                if not present:
                    raise RuntimeError("Required files unavailable; see artifact checks above")
                rc = load_shared(root)
                if args.mode in ("retrieval", "all"):
                    retrieval(rc, args.as_of)
                if args.mode in ("ordering", "all"):
                    ordering(rc, args.as_of, args.item, args.store, args.on_hand)
    except Exception as e:
        show("FAIL", "Validation", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    show("PASS", "Finished", f"{time.perf_counter() - started:.1f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
