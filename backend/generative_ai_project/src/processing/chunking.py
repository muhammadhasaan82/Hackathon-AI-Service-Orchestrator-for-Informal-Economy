"""
Chunking — Converts provider records into rich text for embedding.

Each provider becomes a semantic text chunk optimized for
similarity search with BGE embeddings.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("processing.chunking")


def provider_to_chunk(row: pd.Series) -> str:
    """
    Convert a provider record into a rich text chunk for embedding.

    The text is designed to capture all semantically relevant info
    so that queries like "cheap plumber near Gulberg" match well.
    """
    verified = "Verified" if row.get("verified_provider") else "Unverified"
    languages = row.get("languages_supported", "N/A")

    chunk = (
        f"{row['provider_name']} | {row['category']} | "
        f"{row['area']}, {row['city']} | "
        f"Rating: {row['rating']}/5.0 | "
        f"{row['availability']} | "
        f"{row['experience_years']} years experience | "
        f"{row['completed_jobs']} jobs completed | "
        f"Response time: {row['response_time_minutes']} min | "
        f"Price: {row['price_range']} | "
        f"{verified} | "
        f"Languages: {languages}"
    )
    return chunk


def build_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Build embedding-ready chunks from the full DataFrame.

    Returns list of dicts with:
    - id: provider_id as string
    - text: the chunk text
    - metadata: structured metadata for filtering
    """
    logger.info(f"Building chunks for {len(df)} providers...")

    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "id": str(int(row["provider_id"])),
            "text": provider_to_chunk(row),
            "metadata": {
                "provider_id": int(row["provider_id"]),
                "provider_name": row["provider_name"],
                "category": row["category"],
                "city": row["city"],
                "area": row["area"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "rating": float(row["rating"]),
                "availability": row["availability"],
                "experience_years": int(row["experience_years"]),
                "completed_jobs": int(row["completed_jobs"]),
                "response_time_minutes": int(row["response_time_minutes"]),
                "price_range": row["price_range"],
                "verified_provider": bool(row["verified_provider"]),
                "languages_supported": row["languages_supported"],
                "phone_number": str(row["phone_number"]),
                "email": str(row["email"]),
            },
        })

    logger.info(f"Built {len(chunks)} chunks")
    return chunks
