"""Load Sorare score data from local files or API scaffolding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


SORARE_GRAPHQL_URL = "https://api.sorare.com/graphql"


def load_local_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def sorare_headers() -> dict[str, str]:
    """Build Sorare API headers from environment variables."""
    load_dotenv()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "sorare-score-analysis/1.0",
    }
    jwt = os.getenv("SORARE_JWT")
    jwt_aud = os.getenv("SORARE_JWT_AUD") or os.getenv("JWT_AUD")
    api_key = os.getenv("SORARE_API_KEY") or os.getenv("APIKEY")
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"
    if jwt_aud:
        headers["JWT-AUD"] = jwt_aud
    if api_key:
        headers["APIKEY"] = api_key
    return headers


def run_graphql_query(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a placeholder Sorare GraphQL query.

    This is intentionally generic. Production extraction should add pagination,
    throttling, retries, checkpointing, and query-specific validation.
    """
    response = requests.post(
        SORARE_GRAPHQL_URL,
        json={"query": query, "variables": variables or {}},
        headers=sorare_headers(),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload


def load_data(input_path: str | Path | None = None) -> pd.DataFrame:
    """Load data from a local path.

    API extraction is scaffolded separately via `run_graphql_query`.
    """
    if not input_path:
        raise ValueError("Provide --input with a local CSV or Excel file.")
    return load_local_data(input_path)
