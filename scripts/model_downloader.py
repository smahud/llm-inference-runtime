#!/usr/bin/env python3
"""
Model downloader script for llm-inference-runtime.

Downloads HuggingFace models to cache directory for faster startup.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_model(model_id: str, token: str = "", cache_dir: str = "/root/.cache/huggingface") -> bool:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    try:
        logger.info(f"Downloading model: {model_id}")
        snapshot_download(
            repo_id=model_id,
            token=token if token else None,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True,
        )
        logger.info(f"Model {model_id} downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model for llm-inference-runtime")
    parser.add_argument("--model", required=True, help="Model ID (e.g., meta-llama/Meta-Llama-3-70B-Instruct)")
    parser.add_argument("--token", default="", help="HuggingFace token for private models")
    parser.add_argument("--cache-dir", default="/root/.cache/huggingface", help="Cache directory")
    parser.add_argument("--hf-home", default="", help="HF_HOME environment variable")

    args = parser.parse_args()

    # Set HF_HOME if provided
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = args.cache_dir

    # Get token from env if not provided
    token = args.token or os.environ.get("HF_TOKEN", "")

    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info(f"Model: {args.model}")

    success = download_model(args.model, token, args.cache_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()