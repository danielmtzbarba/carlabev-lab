from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.storage_paths import resolve_artifact_path

if TYPE_CHECKING:
    from src.world_model.evaluate import WorldModelCheckpointEvalResult
    from src.world_model.train import TrainWorldModelResult


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_world_model_results_db_path(path: str) -> Path:
    return resolve_artifact_path(path)


def default_world_model_results_db_path(study_id: str) -> str:
    return f"results/world_model/{study_id.lower()}_runs.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS world_model_study_runs (
            run_name TEXT PRIMARY KEY,
            study_id TEXT NOT NULL,
            exp_id INTEGER NOT NULL,
            seed INTEGER,
            experiment_name TEXT,
            run_dir TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            best_checkpoint_path TEXT NOT NULL,
            epochs INTEGER NOT NULL,
            train_steps INTEGER NOT NULL,
            dataset_paths_json TEXT NOT NULL,
            source_names_json TEXT NOT NULL,
            best_val_loss REAL NOT NULL,
            final_train_loss REAL NOT NULL,
            final_val_loss REAL NOT NULL,
            train_loss REAL,
            train_pred_loss REAL,
            train_reg_loss REAL,
            train_cosine_similarity REAL,
            train_latent_rmse REAL,
            train_explained_variance REAL,
            train_top1_retrieval REAL,
            train_top5_retrieval REAL,
            val_loss REAL NOT NULL,
            val_pred_loss REAL NOT NULL,
            val_reg_loss REAL NOT NULL,
            val_cosine_similarity REAL NOT NULL,
            val_latent_rmse REAL NOT NULL,
            val_explained_variance REAL NOT NULL,
            val_top1_retrieval REAL NOT NULL,
            val_top5_retrieval REAL NOT NULL,
            train_action_metrics_json TEXT,
            train_route_metrics_json TEXT,
            val_action_metrics_json TEXT NOT NULL,
            val_route_metrics_json TEXT NOT NULL,
            eval_output_path TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def record_world_model_study_run(
    *,
    db_path: str,
    study_id: str,
    exp_id: int,
    seed: int | None,
    experiment_name: str | None,
    run_name: str,
    train_result: TrainWorldModelResult,
    eval_result: WorldModelCheckpointEvalResult,
) -> Path:
    resolved_db_path = resolve_world_model_results_db_path(db_path)
    conn = _connect(resolved_db_path)
    try:
        _init_tables(conn)
        now = _utcnow_iso()
        train_metrics = eval_result.train_metrics
        payload = {
            "train_result": asdict(train_result),
            "eval_result": asdict(eval_result),
        }
        conn.execute(
            """
            INSERT INTO world_model_study_runs (
                run_name, study_id, exp_id, seed, experiment_name, run_dir,
                checkpoint_path, best_checkpoint_path, epochs, train_steps,
                dataset_paths_json, source_names_json, best_val_loss,
                final_train_loss, final_val_loss, train_loss, train_pred_loss,
                train_reg_loss, train_cosine_similarity, train_latent_rmse,
                train_explained_variance, train_top1_retrieval, train_top5_retrieval,
                val_loss, val_pred_loss, val_reg_loss, val_cosine_similarity,
                val_latent_rmse, val_explained_variance, val_top1_retrieval,
                val_top5_retrieval, train_action_metrics_json, train_route_metrics_json,
                val_action_metrics_json, val_route_metrics_json, eval_output_path,
                payload_json, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(run_name) DO UPDATE SET
                study_id=excluded.study_id,
                exp_id=excluded.exp_id,
                seed=excluded.seed,
                experiment_name=excluded.experiment_name,
                run_dir=excluded.run_dir,
                checkpoint_path=excluded.checkpoint_path,
                best_checkpoint_path=excluded.best_checkpoint_path,
                epochs=excluded.epochs,
                train_steps=excluded.train_steps,
                dataset_paths_json=excluded.dataset_paths_json,
                source_names_json=excluded.source_names_json,
                best_val_loss=excluded.best_val_loss,
                final_train_loss=excluded.final_train_loss,
                final_val_loss=excluded.final_val_loss,
                train_loss=excluded.train_loss,
                train_pred_loss=excluded.train_pred_loss,
                train_reg_loss=excluded.train_reg_loss,
                train_cosine_similarity=excluded.train_cosine_similarity,
                train_latent_rmse=excluded.train_latent_rmse,
                train_explained_variance=excluded.train_explained_variance,
                train_top1_retrieval=excluded.train_top1_retrieval,
                train_top5_retrieval=excluded.train_top5_retrieval,
                val_loss=excluded.val_loss,
                val_pred_loss=excluded.val_pred_loss,
                val_reg_loss=excluded.val_reg_loss,
                val_cosine_similarity=excluded.val_cosine_similarity,
                val_latent_rmse=excluded.val_latent_rmse,
                val_explained_variance=excluded.val_explained_variance,
                val_top1_retrieval=excluded.val_top1_retrieval,
                val_top5_retrieval=excluded.val_top5_retrieval,
                train_action_metrics_json=excluded.train_action_metrics_json,
                train_route_metrics_json=excluded.train_route_metrics_json,
                val_action_metrics_json=excluded.val_action_metrics_json,
                val_route_metrics_json=excluded.val_route_metrics_json,
                eval_output_path=excluded.eval_output_path,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                run_name,
                study_id,
                exp_id,
                seed,
                experiment_name,
                train_result.run_dir,
                train_result.checkpoint_path,
                train_result.best_checkpoint_path,
                train_result.epochs,
                train_result.train_steps,
                json.dumps(list(eval_result.dataset_paths)),
                json.dumps(list(eval_result.source_names)),
                train_result.best_val_loss,
                train_result.final_train_loss,
                train_result.final_val_loss,
                train_metrics.loss if train_metrics is not None else None,
                train_metrics.pred_loss if train_metrics is not None else None,
                train_metrics.reg_loss if train_metrics is not None else None,
                train_metrics.cosine_similarity if train_metrics is not None else None,
                train_metrics.latent_rmse if train_metrics is not None else None,
                train_metrics.explained_variance if train_metrics is not None else None,
                train_metrics.top1_retrieval if train_metrics is not None else None,
                train_metrics.top5_retrieval if train_metrics is not None else None,
                eval_result.val_metrics.loss,
                eval_result.val_metrics.pred_loss,
                eval_result.val_metrics.reg_loss,
                eval_result.val_metrics.cosine_similarity,
                eval_result.val_metrics.latent_rmse,
                eval_result.val_metrics.explained_variance,
                eval_result.val_metrics.top1_retrieval,
                eval_result.val_metrics.top5_retrieval,
                json.dumps(train_metrics.action_metrics if train_metrics is not None else {}),
                json.dumps(train_metrics.route_metrics if train_metrics is not None else {}),
                json.dumps(eval_result.val_metrics.action_metrics),
                json.dumps(eval_result.val_metrics.route_metrics),
                eval_result.output_path,
                json.dumps(payload),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return resolved_db_path
