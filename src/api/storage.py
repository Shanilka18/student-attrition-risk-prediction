"""TTL-aware storage and cache manager for uploads, predictions, and background jobs.

De-identification Rule:
Does not log student identity fields (study_id). Operates strictly on anonymized
upload_id references and purges files past their configured time-to-live (TTL).
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("api.storage")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class UploadStore:
    """Manages uploaded files, validation reports, and cached scored predictions."""

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._scored_store: Dict[str, Dict[str, Any]] = {}
        self._jobs_store: Dict[str, Dict[str, Any]] = {}

    def cleanup_expired(self) -> int:
        """Purge all expired upload sessions from memory and disk."""
        now = datetime.now(timezone.utc)
        expired_ids = []

        for uid, data in list(self._memory_store.items()):
            expires_at = data.get("expires_at")
            if expires_at and now > expires_at:
                expired_ids.append(uid)

        for uid in expired_ids:
            self._purge_upload(uid)

        # Also purge orphan files on disk older than 2x TTL
        cutoff = now - timedelta(seconds=self.default_ttl * 2)
        disk_purged = 0
        try:
            for f in UPLOAD_DIR.glob("*.*"):
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
                    if mtime < cutoff:
                        f.unlink(missing_ok=True)
                        disk_purged += 1
        except Exception as e:
            logger.warning("Error cleaning disk upload cache: %s", e)

        if expired_ids or disk_purged:
            logger.info("Purged %d expired memory uploads and %d disk files", len(expired_ids), disk_purged)
        return len(expired_ids)

    def _purge_upload(self, upload_id: str):
        """Remove upload session from memory and disk."""
        self._memory_store.pop(upload_id, None)
        self._scored_store.pop(upload_id, None)
        for ext in [".csv", ".parquet", ".json"]:
            path = UPLOAD_DIR / f"{upload_id}{ext}"
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    def save_upload(
        self,
        upload_id: str,
        filename: str,
        df: pd.DataFrame,
        validation_report: Dict[str, Any],
        suggested_mappings: Dict[str, str],
        mapping_confidences: Dict[str, float],
        has_outcomes: bool,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Cache an uploaded dataset and its validation report with an expiry timestamp."""
        self.cleanup_expired()
        ttl = ttl_seconds or self.default_ttl
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl)

        # Save to disk as parquet for lightweight persistence across reloads
        disk_path = UPLOAD_DIR / f"{upload_id}.parquet"
        try:
            df.to_parquet(disk_path, index=False)
        except Exception:
            # Fallback to CSV if parquet engine not available
            disk_path = UPLOAD_DIR / f"{upload_id}.csv"
            df.to_csv(disk_path, index=False)

        entry = {
            "upload_id": upload_id,
            "filename": filename,
            "df": df,
            "validation_report": validation_report,
            "suggested_mappings": suggested_mappings,
            "mapping_confidences": mapping_confidences,
            "has_outcomes": has_outcomes,
            "row_count": len(df),
            "column_count": len(df.columns),
            "detected_columns": df.columns.tolist(),
            "created_at": now,
            "expires_at": expires_at,
            "disk_path": disk_path,
        }
        self._memory_store[upload_id] = entry
        logger.info("Saved upload '%s' (%s, %d rows, TTL: %ds)", upload_id, filename, len(df), ttl)
        return entry

    def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve upload session if not expired."""
        self.cleanup_expired()
        entry = self._memory_store.get(upload_id)
        if not entry:
            # Check disk fallback
            pq_path = UPLOAD_DIR / f"{upload_id}.parquet"
            csv_path = UPLOAD_DIR / f"{upload_id}.csv"
            if pq_path.exists():
                try:
                    df = pd.read_parquet(pq_path)
                    now = datetime.now(timezone.utc)
                    entry = {
                        "upload_id": upload_id,
                        "filename": f"{upload_id}.parquet",
                        "df": df,
                        "validation_report": {},
                        "suggested_mappings": {},
                        "mapping_confidences": {},
                        "has_outcomes": False,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "detected_columns": df.columns.tolist(),
                        "created_at": now,
                        "expires_at": now + timedelta(seconds=self.default_ttl),
                        "disk_path": pq_path,
                    }
                    self._memory_store[upload_id] = entry
                except Exception:
                    return None
            elif csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    now = datetime.now(timezone.utc)
                    entry = {
                        "upload_id": upload_id,
                        "filename": f"{upload_id}.csv",
                        "df": df,
                        "validation_report": {},
                        "suggested_mappings": {},
                        "mapping_confidences": {},
                        "has_outcomes": False,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "detected_columns": df.columns.tolist(),
                        "created_at": now,
                        "expires_at": now + timedelta(seconds=self.default_ttl),
                        "disk_path": csv_path,
                    }
                    self._memory_store[upload_id] = entry
                except Exception:
                    return None
            else:
                return None

        if datetime.now(timezone.utc) > entry["expires_at"]:
            self._purge_upload(upload_id)
            return None

        return entry

    def save_scored_results(
        self,
        upload_id: str,
        cohort_response: Dict[str, Any],
        scored_df: pd.DataFrame,
        fusion_export: Dict[str, Any],
    ) -> None:
        """Cache scored inference predictions for CSV and Fusion JSON download."""
        now = datetime.now(timezone.utc)
        self._scored_store[upload_id] = {
            "cohort_response": cohort_response,
            "scored_df": scored_df,
            "fusion_export": fusion_export,
            "created_at": now,
            "expires_at": now + timedelta(seconds=self.default_ttl),
        }
        # Also cache CSV on disk
        csv_path = UPLOAD_DIR / f"scored_{upload_id}.csv"
        try:
            scored_df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning("Could not persist scored CSV to disk: %s", e)

    def get_scored_results(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve scored prediction artifacts for export."""
        entry = self._scored_store.get(upload_id)
        if not entry:
            csv_path = UPLOAD_DIR / f"scored_{upload_id}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    return {"scored_df": df}
                except Exception:
                    return None
            return None
        return entry

    # =================================================================
    # Retraining Jobs State
    # =================================================================

    def create_job(
        self,
        job_id: str,
        dataset_type: str = "user_dataset",
    ) -> Dict[str, Any]:
        """Initialize tracking entry for a background retraining job."""
        now = datetime.now(timezone.utc).isoformat()
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "progress_pct": 0.0,
            "message": "Job queued for execution",
            "dataset_type": dataset_type,
            "created_at": now,
            "completed_at": None,
            "results": None,
        }
        self._jobs_store[job_id] = job_data
        return job_data

    def update_job(
        self,
        job_id: str,
        status: str,
        progress_pct: float,
        message: str,
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update progress or completion state of a background job."""
        if job_id in self._jobs_store:
            self._jobs_store[job_id]["status"] = status
            self._jobs_store[job_id]["progress_pct"] = round(progress_pct, 1)
            self._jobs_store[job_id]["message"] = message
            if results is not None:
                self._jobs_store[job_id]["results"] = results
            if status in ["completed", "failed"]:
                self._jobs_store[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current status of a background job."""
        return self._jobs_store.get(job_id)


# Global singleton instance
storage = UploadStore()
