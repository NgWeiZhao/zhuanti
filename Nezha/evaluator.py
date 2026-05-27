"""
evaluator.py
Evaluates LLM root cause analysis results against ground truth.
"""

import logging
import datetime

from os.path import dirname
from log import Logger

log_path = dirname(__file__) + '/log/' + str(datetime.datetime.now().strftime(
    '%Y-%m-%d')) + '_nezha.log'
logger = Logger(log_path, logging.DEBUG, __name__).getlog()


def _extract_service_name(pod_name):
    """Extract service name from pod name."""
    service = pod_name.rsplit('-', 1)[0]
    service = service.rsplit('-', 1)[0]
    return service.lower()


def _normalize_fault_type(fault_type):
    """Normalize fault type for flexible matching."""
    ft = fault_type.strip().lower()
    aliases = {
        "cpu": "cpu_contention",
        "cpu_stress": "cpu_contention",
        "high_cpu": "cpu_contention",
        "cpu_overload": "cpu_contention",
        "network": "network_delay",
        "network_latency": "network_delay",
        "net_delay": "network_delay",
        "high_latency": "network_delay",
        "memory_pressure": "memory",
        "memory_leak": "memory",
        "oom": "memory",
        "error": "exception",
        "crash": "exception",
        "bug": "return",
        "logic_error": "return",
        "wrong_return": "return",
    }
    return aliases.get(ft, ft)


def evaluate_single(llm_results, ground_truth_fault):
    """
    Evaluate a single LLM RCA result against ground truth (service-level match).
    Returns rank (1-based) or -1 if not found.
    """
    if not llm_results:
        return -1

    gt_service = _extract_service_name(ground_truth_fault["inject_pod"])

    for i, result in enumerate(llm_results):
        pred_service = result.get("service", "").strip().lower()

        logger.info("  LLM Rank %d: service=%s, fault_type=%s, evidence=%s",
                    i + 1, pred_service, result.get("fault_type", ""),
                    result.get("evidence", "N/A")[:100])

        if pred_service == gt_service:
            logger.info("  -> MATCH at rank %d", i + 1)
            return i + 1

    logger.info("  -> NO MATCH found")
    return -1


def evaluate_single_strict(llm_results, ground_truth_fault):
    """Strict evaluation: both service AND fault_type must match."""
    if not llm_results:
        return -1

    gt_service = _extract_service_name(ground_truth_fault["inject_pod"])
    gt_fault_type = ground_truth_fault["inject_type"].strip().lower()

    for i, result in enumerate(llm_results):
        pred_service = result.get("service", "").strip().lower()
        pred_fault_type = _normalize_fault_type(result.get("fault_type", ""))

        if pred_service == gt_service and pred_fault_type == gt_fault_type:
            return i + 1

    return -1


def compute_accuracy(rank_list, fault_number):
    """Compute Top-K accuracy from a list of ranks."""
    if fault_number == 0:
        return {"top1": 0.0, "top3": 0.0}

    top1 = sum(1 for r in rank_list if r == 1)
    top3 = sum(1 for r in rank_list if 1 <= r <= 3)

    return {
        "top1": top1 / fault_number * 100,
        "top3": top3 / fault_number * 100,
        "top1_count": top1,
        "top3_count": top3,
        "total": fault_number,
        "not_found": sum(1 for r in rank_list if r == -1),
    }


def print_results(accuracy, ns, mode="service"):
    """Print evaluation results."""
    logger.info("=" * 60)
    logger.info("LLM RCA Results (%s, %s level)", ns, mode)
    logger.info("=" * 60)
    logger.info("Total faults: %d", accuracy["total"])
    logger.info("Not found: %d", accuracy["not_found"])
    logger.info("Top-1 Accuracy: %.2f%% (%d/%d)",
                accuracy["top1"], accuracy["top1_count"], accuracy["total"])
    logger.info("Top-3 Accuracy: %.2f%% (%d/%d)",
                accuracy["top3"], accuracy["top3_count"], accuracy["total"])

    print("\n" + "=" * 60)
    print(f"  LLM RCA Results ({ns}, {mode} level)")
    print("=" * 60)
    print(f"  Total faults:   {accuracy['total']}")
    print(f"  Not found:      {accuracy['not_found']}")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy['top1']:.2f}% ({accuracy['top1_count']}/{accuracy['total']})")
    print(f"  Top-3 Accuracy: {accuracy['top3']:.2f}% ({accuracy['top3_count']}/{accuracy['total']})")
    print("=" * 60 + "\n")
