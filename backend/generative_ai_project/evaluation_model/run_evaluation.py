"""
Evaluation Suite — AI Service Orchestrator
============================================
Run the full evaluation suite on GCP.

Usage:
    python evaluation_model/run_evaluation.py --suite all
    python evaluation_model/run_evaluation.py --suite rag
    python evaluation_model/run_evaluation.py --suite llm
    python evaluation_model/run_evaluation.py --suite pipeline
    python evaluation_model/run_evaluation.py --suite reranking

Output:
    evaluation_model/reports/evaluation_report_<timestamp>.json
    evaluation_model/reports/evaluation_report_<timestamp>.html
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from evaluation_model.evaluators.rag_evaluator import RAGEvaluator
from evaluation_model.evaluators.llm_evaluator import LLMEvaluator
from evaluation_model.evaluators.reranking_evaluator import RerankingEvaluator
from evaluation_model.evaluators.pipeline_evaluator import PipelineEvaluator
from evaluation_model.evaluators.booking_evaluator import BookingEvaluator
from evaluation_model.metrics.report_generator import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluation.runner")

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


async def run_suite(suite: str = "all") -> dict:
    """Run the specified evaluation suite and return aggregated results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "suite": suite,
        "scores": {},
        "failures": [],
        "duration_seconds": 0.0,
    }
    start_time = time.time()

    logger.info("═" * 60)
    logger.info(f"  AI SERVICE ORCHESTRATOR — EVALUATION SUITE: {suite.upper()}")
    logger.info("═" * 60)

    # ── RAG Evaluation ──────────────────────────────────────────
    if suite in ("all", "rag"):
        logger.info("\n[1/5] Running RAG Evaluator...")
        try:
            evaluator = RAGEvaluator()
            rag_results = await evaluator.run()
            results["scores"]["rag"] = rag_results
            logger.info(f"  RAG Score: {rag_results.get('overall_score', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"  RAG Evaluator FAILED: {e}")
            results["failures"].append({"suite": "rag", "error": str(e)})

    # ── LLM Evaluation ─────────────────────────────────────────
    if suite in ("all", "llm"):
        logger.info("\n[2/5] Running LLM Evaluator...")
        try:
            evaluator = LLMEvaluator()
            llm_results = await evaluator.run()
            results["scores"]["llm"] = llm_results
            logger.info(f"  LLM Score: {llm_results.get('overall_score', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"  LLM Evaluator FAILED: {e}")
            results["failures"].append({"suite": "llm", "error": str(e)})

    # ── Reranking Evaluation ────────────────────────────────────
    if suite in ("all", "reranking"):
        logger.info("\n[3/5] Running Reranking Evaluator...")
        try:
            evaluator = RerankingEvaluator()
            rerank_results = evaluator.run()
            results["scores"]["reranking"] = rerank_results
            logger.info(f"  Reranking NDCG@10: {rerank_results.get('ndcg_at_10', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"  Reranking Evaluator FAILED: {e}")
            results["failures"].append({"suite": "reranking", "error": str(e)})

    # ── Booking Evaluation ─────────────────────────────────────
    if suite in ("all", "booking"):
        logger.info("\n[4/5] Running Booking Evaluator...")
        try:
            evaluator = BookingEvaluator()
            booking_results = await evaluator.run()
            results["scores"]["booking"] = booking_results
            logger.info(f"  Booking Accuracy: {booking_results.get('accuracy', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"  Booking Evaluator FAILED: {e}")
            results["failures"].append({"suite": "booking", "error": str(e)})

    # ── End-to-End Pipeline Evaluation ─────────────────────────
    if suite in ("all", "pipeline"):
        logger.info("\n[5/5] Running Pipeline Evaluator...")
        try:
            evaluator = PipelineEvaluator()
            pipeline_results = await evaluator.run()
            results["scores"]["pipeline"] = pipeline_results
            logger.info(f"  Pipeline Success Rate: {pipeline_results.get('success_rate', 'N/A'):.3f}")
        except Exception as e:
            logger.error(f"  Pipeline Evaluator FAILED: {e}")
            results["failures"].append({"suite": "pipeline", "error": str(e)})

    # ── Aggregate ──────────────────────────────────────────────
    results["duration_seconds"] = time.time() - start_time
    sub_scores = [v.get("overall_score", 0) for v in results["scores"].values() if isinstance(v, dict)]
    results["aggregate_score"] = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0

    # ── Reports ────────────────────────────────────────────────
    report_path_json = REPORTS_DIR / f"evaluation_report_{timestamp}.json"
    report_path_html = REPORTS_DIR / f"evaluation_report_{timestamp}.html"

    with open(report_path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results, str(report_path_html))

    logger.info("\n" + "═" * 60)
    logger.info(f"  AGGREGATE SCORE: {results['aggregate_score']:.3f}")
    logger.info(f"  Duration:        {results['duration_seconds']:.1f}s")
    logger.info(f"  Failures:        {len(results['failures'])}")
    logger.info(f"  Report (JSON):   {report_path_json}")
    logger.info(f"  Report (HTML):   {report_path_html}")
    logger.info("═" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Service Orchestrator Evaluation Suite")
    parser.add_argument(
        "--suite",
        choices=["all", "rag", "llm", "reranking", "booking", "pipeline"],
        default="all",
        help="Which evaluation suite to run",
    )
    args = parser.parse_args()
    asyncio.run(run_suite(args.suite))
