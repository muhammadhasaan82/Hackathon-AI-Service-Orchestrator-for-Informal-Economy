"""
Build Embeddings — Standalone script to index all providers.
Run: python -m scripts.build_embeddings
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from src.rag.indexer import index_providers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index service providers into vector store")
    parser.add_argument("--force", action="store_true", help="Force rebuild embeddings")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    args = parser.parse_args()

    store = index_providers(csv_path=args.csv, force_rebuild=args.force)
    print(f"\n✅ Indexing complete: {store.count} providers in vector store")
