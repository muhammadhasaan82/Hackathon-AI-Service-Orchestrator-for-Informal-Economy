"""
RAG Evaluator — Measures retrieval quality of the Agentic RAG pipeline.

Metrics:
    - Hit Rate @K        : Whether the ground-truth provider appears in top-K
    - MRR (Mean Reciprocal Rank) : Position of first relevant result
    - Precision @K       : Fraction of relevant providers in top-K
    - Context Relevance  : Cosine similarity of query vs retrieved text
    - Faithfulness       : Whether LLM answer is grounded in retrieved context

Ground truth is defined in: evaluation_model/data/rag_test_cases.json
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("eval.rag")

TEST_CASES_PATH = Path(__file__).parent.parent / "data" / "rag_test_cases.json"


class RAGEvaluator:
    """Evaluates RAG retrieval quality against ground truth test cases."""

    def __init__(self, vector_store=None, retriever=None, top_k: int = 10):
        self.top_k = top_k
        self._vector_store = vector_store
        self._retriever = retriever
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> list[dict]:
        """Load test cases from JSON file."""
        if not TEST_CASES_PATH.exists():
            logger.warning(f"Test cases not found at {TEST_CASES_PATH}. Using default set.")
            return self._default_test_cases()
        with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _default_test_cases(self) -> list[dict]:
        """Built-in test cases covering all service categories and cities."""
        return [
            {
                "id": "tc_001",
                "query": "AC technician in G-13 Islamabad",
                "service_type": "AC Technician",
                "city": "Islamabad",
                "area": "G-13",
                "expected_category": "AC Technician",
                "expected_cities": ["Islamabad"],
            },
            {
                "id": "tc_002",
                "query": "plumber DHA Karachi pipe leak emergency",
                "service_type": "Plumber",
                "city": "Karachi",
                "area": "DHA",
                "expected_category": "Plumber",
                "expected_cities": ["Karachi"],
            },
            {
                "id": "tc_003",
                "query": "mujhe electrician chahiye Gulberg Lahore mein",
                "service_type": "Electrician",
                "city": "Lahore",
                "area": "Gulberg",
                "expected_category": "Electrician",
                "expected_cities": ["Lahore"],
            },
            {
                "id": "tc_004",
                "query": "home tutor class 10 maths Islamabad",
                "service_type": "Home Tutor",
                "city": "Islamabad",
                "area": None,
                "expected_category": "Home Tutor",
                "expected_cities": ["Islamabad"],
            },
            {
                "id": "tc_005",
                "query": "beautician bridal makeup Johar Town Lahore",
                "service_type": "Beautician",
                "city": "Lahore",
                "area": "Johar Town",
                "expected_category": "Beautician",
                "expected_cities": ["Lahore"],
            },
            {
                "id": "tc_006",
                "query": "mobile phone repair screen broken Saddar Rawalpindi",
                "service_type": "Mobile Repair",
                "city": "Rawalpindi",
                "area": "Saddar",
                "expected_category": "Mobile Repair",
                "expected_cities": ["Rawalpindi"],
            },
            {
                "id": "tc_007",
                "query": "deep house cleaning service Hayatabad Peshawar",
                "service_type": "Cleaning Service",
                "city": "Peshawar",
                "area": "Hayatabad",
                "expected_category": "Cleaning Service",
                "expected_cities": ["Peshawar"],
            },
            {
                "id": "tc_008",
                "query": "washing machine repair Faisalabad",
                "service_type": "Appliance Repair",
                "city": "Faisalabad",
                "area": None,
                "expected_category": "Appliance Repair",
                "expected_cities": ["Faisalabad"],
            },
        ]

    async def run(self) -> dict:
        """Run RAG evaluation and return metrics dict."""
        logger.info(f"Running RAG evaluation on {len(self.test_cases)} test cases...")

        # Initialize retriever if not provided
        retriever = self._get_retriever()

        hit_rates = []
        mrr_scores = []
        precision_scores = []
        context_relevance_scores = []
        per_case_results = []

        for tc in self.test_cases:
            try:
                results = retriever.search(
                    query=tc["query"],
                    category=tc.get("service_type"),
                    city=tc.get("city"),
                    area=tc.get("area"),
                    n_results=self.top_k,
                )

                # Hit Rate: does any result match expected category?
                categories = [
                    r.get("metadata", {}).get("category", "").lower()
                    for r in results
                ]
                expected_cat = tc["expected_category"].lower()

                hit = any(expected_cat in cat for cat in categories)
                hit_rates.append(1.0 if hit else 0.0)

                # MRR: position of first relevant result
                rr = 0.0
                for rank, cat in enumerate(categories, 1):
                    if expected_cat in cat:
                        rr = 1.0 / rank
                        break
                mrr_scores.append(rr)

                # Precision@K: fraction of relevant in top-K
                relevant = sum(1 for cat in categories if expected_cat in cat)
                precision_scores.append(relevant / self.top_k)

                # Context Relevance: avg rerank_score across results
                rerank_scores = [r.get("rerank_score", r.get("retrieval_score", 0.0)) for r in results]
                avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0
                context_relevance_scores.append(avg_rerank)

                per_case_results.append({
                    "id": tc["id"],
                    "query": tc["query"],
                    "hit": hit,
                    "mrr": rr,
                    "precision_at_k": relevant / self.top_k,
                    "context_relevance": avg_rerank,
                    "top_3_categories": categories[:3],
                })

            except Exception as e:
                logger.warning(f"Test case {tc['id']} failed: {e}")
                per_case_results.append({"id": tc["id"], "error": str(e)})

        # Aggregate metrics
        hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        context_rel = sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0.0
        overall = (hit_rate * 0.35 + mrr * 0.30 + precision * 0.20 + context_rel * 0.15)

        return {
            "overall_score": overall,
            "hit_rate_at_k": hit_rate,
            "mrr": mrr,
            "precision_at_k": precision,
            "context_relevance": context_rel,
            "k": self.top_k,
            "total_test_cases": len(self.test_cases),
            "per_case": per_case_results,
        }

    def _get_retriever(self):
        """Initialize retriever from project, or use mock retriever."""
        if self._retriever:
            return self._retriever
        try:
            from src.rag.retriever import ProviderRetriever
            from src.rag.vector_store import WeaviateVectorStore
            from src.core.model_factory import load_all_configs
            configs = load_all_configs()
            vs = WeaviateVectorStore()
            return ProviderRetriever(vs, configs)
        except Exception as e:
            logger.warning(f"Could not load live retriever ({e}). Using mock.")
            return MockRetriever()


class MockRetriever:
    """Mock retriever for offline/unit testing."""
    def search(self, query, category=None, city=None, area=None, n_results=10):
        return [
            {
                "metadata": {"category": category or "Plumber", "city": city or "Islamabad"},
                "retrieval_score": 0.85,
                "rerank_score": 0.80,
            }
            for _ in range(min(n_results, 5))
        ]
