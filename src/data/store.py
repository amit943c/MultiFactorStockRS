"""Parquet-based caching layer for price and factor DataFrames.

Saves artefacts under ``data/processed/`` and performs a simple
time-based staleness check so that downstream code can skip expensive
re-fetches when fresh data is already on disk.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "processed"
_DEFAULT_MAX_AGE_SECONDS: float = 24 * 60 * 60  # 24 hours


class DataStore:
    """Read / write DataFrames as Parquet files with optional staleness checks.

    Parameters
    ----------
    cache_dir:
        Directory for cached parquet files.  Defaults to
        ``<project_root>/data/processed/``.
    max_age_seconds:
        Maximum age (in seconds) before a cached file is considered stale.
        Defaults to 24 hours.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_age_seconds: float = _DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_age = max_age_seconds
        logger.info("DataStore initialised — cache_dir=%s, max_age=%ss", self._cache_dir, self._max_age)

    # ------------------------------------------------------------------
    # Public API — prices
    # ------------------------------------------------------------------

    def save_prices(self, df: pd.DataFrame, name: str = "prices") -> Path:
        """Persist a price DataFrame to Parquet.

        Returns the path of the written file.
        """
        return self._write(df, name)

    def load_prices(self, name: str = "prices") -> pd.DataFrame:
        """Load a previously cached price DataFrame.

        Raises
        ------
        FileNotFoundError
            If no cached file exists for *name*.
        """
        return self._read(name)

    # ------------------------------------------------------------------
    # Public API — factors
    # ------------------------------------------------------------------

    def save_factors(self, df: pd.DataFrame, name: str = "factors") -> Path:
        """Persist a factor DataFrame to Parquet."""
        return self._write(df, name)

    def load_factors(self, name: str = "factors") -> pd.DataFrame:
        """Load a previously cached factor DataFrame.

        Raises
        ------
        FileNotFoundError
            If no cached file exists for *name*.
        """
        return self._read(name)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def has_cache(self, name: str) -> bool:
        """Return ``True`` if a non-stale cache file exists for *name*."""
        path = self._path_for(name)
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        if age > self._max_age:
            logger.info("Cache for %r is stale (%.1f h old)", name, age / 3600)
            return False
        return True

    def clear_cache(self, name: str | None = None) -> None:
        """Delete cached Parquet file(s).

        Parameters
        ----------
        name:
            If given, delete only that file.  Otherwise remove **all**
            ``.parquet`` files in the cache directory.
        """
        if name is not None:
            path = self._path_for(name)
            if path.exists():
                path.unlink()
                logger.info("Deleted cache file %s", path)
            return

        removed = 0
        for pq in self._cache_dir.glob("*.parquet"):
            pq.unlink()
            removed += 1
        logger.info("Cleared %d cached file(s) from %s", removed, self._cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def _write(self, df: pd.DataFrame, name: str) -> Path:
        path = self._path_for(name)
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
        return path

    def _read(self, name: str) -> pd.DataFrame:
        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"No cached file at {path}")
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
