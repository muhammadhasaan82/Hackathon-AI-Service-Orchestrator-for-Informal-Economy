"""
Preprocessor — CSV data loading, cleaning, and normalization.

Loads the 50K service provider dataset and prepares it for
embedding generation and vector store indexing.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("processing.preprocessor")

# Default dataset path — relative to project root
_DEFAULT_CSV = Path(__file__).parent.parent.parent.parent / "service_providers_50000.csv"


def load_providers(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and clean the service providers dataset.

    Returns a normalized DataFrame with consistent types.
    """
    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    logger.info(f"Loading providers from: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} raw records")

    # ── Normalize columns ───────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── Type casting ────────────────────────────────────────
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce").fillna(0).astype(int)
    df["completed_jobs"] = pd.to_numeric(df["completed_jobs"], errors="coerce").fillna(0).astype(int)
    df["response_time_minutes"] = pd.to_numeric(df["response_time_minutes"], errors="coerce").fillna(60).astype(int)
    df["verified_provider"] = df["verified_provider"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    df["provider_id"] = df["provider_id"].astype(int)

    # ── Standardize categories ──────────────────────────────
    df["category"] = df["category"].str.title()
    # Fix acronyms that str.title() breaks
    _acronym_fixes = {"Ac ": "AC ", "Tv ": "TV "}
    for wrong, right in _acronym_fixes.items():
        df["category"] = df["category"].str.replace(wrong, right, regex=False)
    df["city"] = df["city"].str.title()
    df["area"] = df["area"].str.strip()

    # Drop rows with missing critical fields
    critical = ["provider_id", "provider_name", "category", "city", "latitude", "longitude"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing critical fields")

    logger.info(f"Preprocessed {len(df)} providers | {df['category'].nunique()} categories | {df['city'].nunique()} cities")
    return df


def get_service_categories(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique service categories."""
    return sorted(df["category"].unique().tolist())


def get_cities(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique cities."""
    return sorted(df["city"].unique().tolist())


def get_areas_by_city(df: pd.DataFrame) -> dict[str, list[str]]:
    """Get areas grouped by city."""
    return {
        city: sorted(group["area"].unique().tolist())
        for city, group in df.groupby("city")
    }


def provider_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a clean dictionary."""
    return {
        "provider_id": int(row["provider_id"]),
        "provider_name": row["provider_name"],
        "category": row["category"],
        "city": row["city"],
        "area": row["area"],
        "full_location": row.get("full_location", f"{row['area']}, {row['city']}"),
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
        "phone_number": row["phone_number"],
        "email": row["email"],
    }
