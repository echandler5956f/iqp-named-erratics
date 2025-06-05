import os
import hashlib
import time
import logging
import json
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

import portalocker  # cross-platform file locking for concurrency-safe downloads

logger = logging.getLogger(__name__)


class RawDownloadCache:
    """Shared on-disk cache for **raw** remote datasets.

    Remote resources (HTTP/HTTPS/FTP) are downloaded once per URL and reused by
    all subsequent pipeline invocations – even across parallel processes.  Each
    resource lives under a deterministic path containing a truncated SHA-256 of
    its URL, so differing query-strings or mirrors do not collide.
    """

    _ENV_VAR = "IQP_DATA_CACHE"

    def __init__(self, root_dir: Optional[str] = None):
        # Determine cache root using precedence: explicit arg → environment → repo default
        if root_dir is None:
            root_dir = os.getenv(self._ENV_VAR)
        if root_dir is None:
            module_dir = Path(__file__).resolve().parent
            root_dir = module_dir.parent / "data" / "raw_cache"
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("RawDownloadCache root set to %s", self.root_dir)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def target_path(self, url: str, source_name: str) -> Path:
        """Return canonical cache path for *url*."""
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "data"
        filename = filename.split("?")[0]  # strip query strings that appear in filename
        sha16 = hashlib.sha256(url.encode()).hexdigest()[:16]
        subdir = self.root_dir / source_name / sha16
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / filename

    def ensure(self, url: str, source_name: str, download_fn: Callable[[str], bool]) -> Optional[str]:
        """Ensure *url* is downloaded.

        Parameters
        ----------
        url : str
            Remote URL pointing to the dataset.
        source_name : str
            Unique DataSource name for grouping under cache root.
        download_fn : Callable[[str], bool]
            Function that performs the actual download **to the given file path**.
            Must return True on success.
        """
        dest_path = self.target_path(url, source_name)
        lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
        meta_path = dest_path.with_suffix(dest_path.suffix + ".meta.json")

        # Fast path: file already present and non-empty.
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return str(dest_path)

        # Acquire exclusive lock to avoid duplicate downloads.
        try:
            with portalocker.Lock(str(lock_path), timeout=1800):  # 30-minute safety timeout
                # Re-check after obtaining lock.
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    return str(dest_path)

                logger.info("Downloading remote resource for '%s' → %s", source_name, dest_path)
                success = download_fn(str(dest_path))
                if not success or not dest_path.exists():
                    logger.error("Download failed for %s", url)
                    # Clean up potential partial file
                    if dest_path.exists():
                        try:
                            dest_path.unlink()
                        except OSError:
                            pass
                    return None

                # Write metadata (best effort)
                try:
                    meta = {
                        "url": url,
                        "downloaded_at": int(time.time()),
                        "size_bytes": dest_path.stat().st_size,
                    }
                    with open(meta_path, "w", encoding="utf-8") as fp:
                        json.dump(meta, fp)
                except Exception as meta_err:  # pragma: no cover
                    logger.debug("Failed to write metadata for %s: %s", dest_path, meta_err)

                return str(dest_path)
        except portalocker.exceptions.LockException as exc:
            logger.error("Timeout waiting to acquire lock for %s: %s", dest_path, exc)
            return None 