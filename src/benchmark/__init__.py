"""再現可能なRAGベンチマークのデータ構造と純粋な評価処理。"""

from src.benchmark.dataset import BenchmarkDatasetError, load_benchmark_cases
from src.benchmark.metrics import (
    answerability_accuracy,
    compute_metrics,
    confusion_matrix,
    distance_false_accept_rate,
    distance_false_reject_rate,
    distance_gate_accuracy,
    false_answer_rate,
    false_refusal_rate,
    mean_reciprocal_rank_at_k,
    retrieval_hit_rate_at_k,
    status_accuracy,
)
from src.benchmark.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkCategory,
    BenchmarkExpectation,
    BenchmarkMetrics,
    BenchmarkOptions,
    BenchmarkSplit,
    GoldSource,
    RetrievedSource,
)
from src.benchmark.runner import BenchmarkQAService, run_benchmark

__all__ = (
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkCategory",
    "BenchmarkDatasetError",
    "BenchmarkExpectation",
    "BenchmarkMetrics",
    "BenchmarkOptions",
    "BenchmarkQAService",
    "BenchmarkSplit",
    "GoldSource",
    "RetrievedSource",
    "answerability_accuracy",
    "compute_metrics",
    "confusion_matrix",
    "distance_false_accept_rate",
    "distance_false_reject_rate",
    "distance_gate_accuracy",
    "false_answer_rate",
    "false_refusal_rate",
    "load_benchmark_cases",
    "mean_reciprocal_rank_at_k",
    "retrieval_hit_rate_at_k",
    "run_benchmark",
    "status_accuracy",
)
