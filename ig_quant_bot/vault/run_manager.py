from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yaml


class RunManager:
    """
    V7.2 Research Vault & Provenance Manager.

    Persists:
      - ledger.parquet
      - fills.parquet
      - trades.parquet
      - config_used.yaml
      - metadata.json (run manifest)

    This version accepts metadata as a dict and keeps dataframe writes explicit.
    """

    def __init__(self, base_path: str = "vault/runs"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def persist_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        *,
        ledger_df: Optional[pd.DataFrame] = None,
        fills_df: Optional[pd.DataFrame] = None,
        trades_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        run_dir = os.path.join(self.base_path, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save parquet artifacts
        if ledger_df is not None and not ledger_df.empty:
            ledger_df.to_parquet(os.path.join(run_dir, "ledger.parquet"))
        if fills_df is not None and not fills_df.empty:
            fills_df.to_parquet(os.path.join(run_dir, "fills.parquet"))
        if trades_df is not None and not trades_df.empty:
            trades_df.to_parquet(os.path.join(run_dir, "trades.parquet"))

        # Config snapshot
        with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Manifest
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_snapshot": config,
            "metadata": metadata or {},
            "summary": {
                "total_fills": int(len(fills_df)) if fills_df is not None else 0,
                "total_trades": int(len(trades_df)) if trades_df is not None else 0,
                "final_equity": float(ledger_df["equity"].iloc[-1]) if ledger_df is not None and ("equity" in ledger_df.columns) and (not ledger_df.empty) else 0.0,
            },
        }

        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"[*] Research artifacts persisted to {run_dir}")

    def load_run(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.base_path, run_id)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run ID {run_id} not found in vault.")

        out: Dict[str, Any] = {}
        for name in ["ledger", "fills", "trades"]:
            p = os.path.join(run_dir, f"{name}.parquet")
            out[name] = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

        meta_p = os.path.join(run_dir, "metadata.json")
        out["metadata"] = json.load(open(meta_p)) if os.path.exists(meta_p) else {}

        return out

    def list_runs(self) -> list:
        return sorted(os.listdir(self.base_path), reverse=True)
