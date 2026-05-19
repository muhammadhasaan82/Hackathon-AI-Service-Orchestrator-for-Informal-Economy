"""
Reranking Evaluator — Measures precision improvement from cross-encoder reranking.

Metrics: NDCG@K, MAP, Delta improvement over vector-only baseline.
"""

import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("eval.reranking")

RERANKING_TEST_CASES = [
    {
        "id": "rk_001",
        "query": "urgent AC repair today Islamabad",
        "candidates": [
            {"text": "AC Technician G-13 Islamabad, 5 years experience, rating 4.8", "relevance": 3},
            {"text": "Plumber F-10 Islamabad water heater", "relevance": 0},
            {"text": "AC Technician I-8 Islamabad, same day service", "relevance": 3},
            {"text": "Electrician Rawalpindi available now", "relevance": 0},
            {"text": "AC Technician Islamabad gas refill", "relevance": 2},
            {"text": "Home cleaner Islamabad affordable", "relevance": 0},
        ],
    },
    {
        "id": "rk_002",
        "query": "cheap plumber pipe leak DHA Karachi",
        "candidates": [
            {"text": "Plumber DHA Karachi budget rates 10 years", "relevance": 3},
            {"text": "Electrician DHA Karachi certified", "relevance": 0},
            {"text": "Plumber Clifton Karachi emergency", "relevance": 2},
            {"text": "Plumber near DHA Karachi affordable", "relevance": 3},
            {"text": "Painter DHA Karachi", "relevance": 0},
            {"text": "Plumber Karachi 24/7 available", "relevance": 1},
        ],
    },
    {
        "id": "rk_003",
        "query": "bridal beautician Lahore Saturday",
        "candidates": [
            {"text": "Beautician Lahore bridal specialist model town", "relevance": 2},
            {"text": "Beautician Gulberg Lahore bridal packages", "relevance": 3},
            {"text": "Tutor Lahore O-level chemistry", "relevance": 0},
            {"text": "Beautician Johar Town Lahore makeup artist", "relevance": 3},
            {"text": "Carpenter Lahore furniture", "relevance": 0},
            {"text": "Beautician Lahore general services", "relevance": 1},
        ],
    },
]


def _ndcg(relevances: list, k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(relevances, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(relevances: list) -> float:
    hits, total = 0, 0.0
    for rank, rel in enumerate(relevances, 1):
        if rel > 0:
            hits += 1
            total += hits / rank
    return total / hits if hits > 0 else 0.0


class RerankingEvaluator:
    """Evaluates cross-encoder reranker via NDCG and MAP."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu", k: int = 5):
        self.model_name = model_name
        self.device = device
        self.k = k

    def run(self) -> dict:
        logger.info(f"Running Reranking evaluation on {len(RERANKING_TEST_CASES)} cases...")
        try:
            from src.rag.reranker import rerank
            reranker_available = True
        except Exception as e:
            logger.warning(f"Reranker not available: {e}. Using mock.")
            reranker_available = False

        ndcg_before, ndcg_after, map_before, map_after = [], [], [], []
        per_case = []

        for tc in RERANKING_TEST_CASES:
            try:
                cands = [dict(c) for c in tc["candidates"]]
                gt = [c["relevance"] for c in tc["candidates"]]

                nb = _ndcg(gt, self.k)
                ab = _average_precision(gt)
                ndcg_before.append(nb)
                map_before.append(ab)

                if reranker_available:
                    reranked = rerank(tc["query"], cands, text_key="text",
                                      top_n=len(cands), model_name=self.model_name, device=self.device)
                else:
                    reranked = sorted(cands, key=lambda x: len(x["text"]), reverse=True)

                reranked_gt = [
                    next((c["relevance"] for c in tc["candidates"] if c["text"] == r["text"]), 0)
                    for r in reranked
                ]
                na = _ndcg(reranked_gt, self.k)
                aa = _average_precision(reranked_gt)
                ndcg_after.append(na)
                map_after.append(aa)

                per_case.append({"id": tc["id"], "ndcg_before": nb, "ndcg_after": na,
                                  "delta": na - nb, "map_before": ab, "map_after": aa})
            except Exception as e:
                logger.warning(f"{tc['id']} failed: {e}")
                per_case.append({"id": tc["id"], "error": str(e)})

        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        na = avg(ndcg_after)
        nb = avg(ndcg_before)

        return {
            "overall_score": na,
            "ndcg_at_k": na,
            "ndcg_at_k_baseline": nb,
            "ndcg_improvement": na - nb,
            "map_after": avg(map_after),
            "map_baseline": avg(map_before),
            "k": self.k,
            "total_test_cases": len(RERANKING_TEST_CASES),
            "per_case": per_case,
        }
