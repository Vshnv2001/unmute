"""
Google Cloud Storage helper for accessing datasets.

This module provides a unified interface for accessing dataset files,
supporting both local filesystem (for development) and GCS (for production).

Environment Variables:
    GCS_BUCKET_NAME: Name of the GCS bucket (e.g., 'unmute-datasets')
    USE_GCS: Set to 'true' to use GCS, otherwise uses local filesystem
"""

import os
import io
import pickle
import json
from typing import Optional, Any
from functools import lru_cache

# Check if we should use GCS
USE_GCS = os.environ.get("USE_GCS", "false").lower() == "true"
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "unmute-datasets")

# GCS public URL base (for static file serving)
GCS_PUBLIC_URL = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}"

# Initialize GCS client only if needed
_gcs_client = None
_gcs_bucket = None


def _get_gcs_bucket():
    """Lazily initialize GCS client and bucket."""
    global _gcs_client, _gcs_bucket
    if _gcs_bucket is None and USE_GCS:
        try:
            from google.cloud import storage
            _gcs_client = storage.Client()
            _gcs_bucket = _gcs_client.bucket(GCS_BUCKET_NAME)
            print(f"[GCS] Connected to bucket: {GCS_BUCKET_NAME}")
        except Exception as e:
            print(f"[GCS] Failed to connect to bucket: {e}")
            raise
    return _gcs_bucket


def get_static_url(relative_path: str) -> str:
    """
    Get the URL for a static file.
    
    Args:
        relative_path: Path relative to the dataset root (e.g., 'sgsl_dataset/HELLO/HELLO.gif')
    
    Returns:
        URL to access the file (either local /static/ path or GCS public URL)
    """
    if USE_GCS:
        # Return GCS public URL
        return f"{GCS_PUBLIC_URL}/{relative_path}"
    else:
        # Return local static path
        return f"/static/{relative_path}"


def file_exists(relative_path: str, local_base_dir: str = None) -> bool:
    """
    Check if a file exists (locally or in GCS).
    
    Args:
        relative_path: Path relative to the dataset root
        local_base_dir: Base directory for local files (used when USE_GCS=false)
    """
    if USE_GCS:
        try:
            bucket = _get_gcs_bucket()
            blob = bucket.blob(relative_path)
            return blob.exists()
        except Exception as e:
            print(f"[GCS] Error checking file existence: {e}")
            return False
    else:
        if local_base_dir:
            full_path = os.path.join(local_base_dir, relative_path)
        else:
            # Default to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, relative_path)
        return os.path.exists(full_path)


def read_json(relative_path: str, local_base_dir: str = None) -> Optional[dict]:
    """
    Read a JSON file from GCS or local filesystem.
    
    Args:
        relative_path: Path relative to the dataset root (e.g., 'sgsl_processed/vocab.json')
        local_base_dir: Base directory for local files
    
    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    if USE_GCS:
        try:
            bucket = _get_gcs_bucket()
            blob = bucket.blob(relative_path)
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            print(f"[GCS] Error reading JSON {relative_path}: {e}")
            return None
    else:
        if local_base_dir:
            full_path = os.path.join(local_base_dir, relative_path)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, relative_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


def read_pickle(relative_path: str, local_base_dir: str = None) -> Optional[Any]:
    """
    Read a pickle file from GCS or local filesystem.
    
    Args:
        relative_path: Path relative to the dataset root (e.g., 'sgsl_processed/landmarks_pkl/HELLO.pkl')
        local_base_dir: Base directory for local files
    
    Returns:
        Unpickled data or None if file doesn't exist
    """
    if USE_GCS:
        try:
            bucket = _get_gcs_bucket()
            blob = bucket.blob(relative_path)
            content = blob.download_as_bytes()
            return pickle.loads(content)
        except Exception as e:
            print(f"[GCS] Error reading pickle {relative_path}: {e}")
            return None
    else:
        if local_base_dir:
            full_path = os.path.join(local_base_dir, relative_path)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, relative_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        return None


@lru_cache(maxsize=100)
def read_pickle_cached(relative_path: str) -> Optional[Any]:
    """
    Read a pickle file with caching (for frequently accessed files).
    
    Note: Uses LRU cache to avoid repeated GCS calls for the same file.
    """
    return read_pickle(relative_path)


def get_dataset_info() -> dict:
    """Get information about the current storage configuration."""
    return {
        "use_gcs": USE_GCS,
        "bucket_name": GCS_BUCKET_NAME if USE_GCS else None,
        "public_url": GCS_PUBLIC_URL if USE_GCS else None,
    }
