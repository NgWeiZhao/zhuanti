"""
main_llm.py
Main entry point for LLM-based root cause analysis on Nezha dataset.

Usage:
    # Step 1: Set API key (PowerShell):
    $env:GEMINI_API_KEY="your_key_here"

    # Step 2: Run:
    python main_llm.py --ns hipster --provider gemini
    python main_llm.py --ns hipster --provider gemini --compare-normal
    python main_llm.py --ns ts --provider gemini
"""

import argparse
import json
import time
import datetime
import logging

from os.path import dirname
from log import Logger
from data_collector import collect_profiling_data
from prompt_builder import build_rca_prompt, build_rca_prompt_with_normal
from llm_ranker import llm_root_cause_analysis
from evaluator import evaluate_single, compute_accuracy, print_results

log_path = dirname(__file__) + '/log/' + str(datetime.datetime.now().strftime(
    '%Y-%m-%d')) + '_nezha_llm.log'
logger = Logger(log_path, logging.DEBUG, __name__).getlog()


def run_evaluation(ns, provider, model_name, compare_normal, api_key=None, delay=4.0):
    """Run LLM-based RCA evaluation on the Nezha dataset."""
    rca_data_path = dirname(__file__) + "/rca_data"
    construction_data_path = dirname(__file__) + "/construct_data"

    # Dataset configuration (same as original main.py)
    if ns == "hipster":
        fault_inject_files = [
            rca_data_path + "/2022-08-22/2022-08-22-fault_list.json",
            rca_data_path + "/2022-08-23/2022-08-23-fault_list.json",
        ]
        normal_times = ["2022-08-22 03:51", "2022-08-23 17:00"]
    elif ns == "ts":
        fault_inject_files = [
            rca_data_path + "/2023-01-29/2023-01-29-fault_list.json",
            rca_data_path + "/2023-01-30/2023-01-30-fault_list.json",
        ]
        normal_times = ["2023-01-29 08:50", "2023-01-30 11:39"]
    else:
        print(f"Unknown namespace: {ns}")
        return

    # Collect normal baseline data if needed
    normal_profiling_data_list = []
    if compare_normal:
        for normal_time in normal_times:
            print(f"[INFO] Collecting normal baseline data for {normal_time}...")
            normal_data = collect_profiling_data(normal_time, ns, construction_data_path)
            normal_profiling_data_list.append(normal_data)

    # Count total faults first for progress display
    total_faults = 0
    for fault_file in fault_inject_files:
        with open(fault_file, 'r') as f:
            fd = json.load(f)
        for hk in fd:
            total_faults += len(fd[hk])
    print(f"[INFO] Total faults to evaluate: {total_faults}")
    print(f"[INFO] Estimated time: ~{int(total_faults * (delay + 40))} seconds ({total_faults * (delay + 40) / 60:.1f} min)\n")

    # Iterate through all fault cases
    rank_list = []
    fault_number = 0
    detailed_results = []

    for file_idx, fault_file in enumerate(fault_inject_files):
        print(f"\n[INFO] Processing fault file: {fault_file}")
        with open(fault_file, 'r') as f:
            fault_inject_data = json.load(f)

        normal_profiling_data = normal_profiling_data_list[file_idx] if compare_normal else None

        for hour_key in fault_inject_data:
            for fault in fault_inject_data[hour_key]:
                fault_number += 1
                inject_time = fault["inject_time"]
                inject_pod = fault["inject_pod"]
                inject_type = fault["inject_type"]

                print(f"\n{'='*60}")
                print(f"[Fault {fault_number}/{total_faults}] Time: {inject_time}")
                print(f"  Ground Truth: pod={inject_pod}, type={inject_type}")
                print(f"{'='*60}")

                # Calculate abnormal time (inject_time + 2 minutes, same as original)
                min_val = int(inject_time.split(":")[1]) + 2
                if min_val >= 60:
                    hour_min = inject_time.split(" ")[1]
                    hour_val = int(hour_min.split(":")[0])
                    if hour_val < 9:
                        abnormal_time = inject_time.split(" ")[0] + " 0" + str(hour_val + 1) + ":0" + str(min_val - 60)
                    else:
                        abnormal_time = inject_time.split(" ")[0] + " " + str(hour_val + 1) + ":0" + str(min_val - 60)
                elif min_val < 10:
                    abnormal_time = inject_time.split(":")[0] + ":0" + str(min_val)
                else:
                    abnormal_time = inject_time.split(":")[0] + ":" + str(min_val)

                # Remove seconds if present
                parts = abnormal_time.split(":")
                if len(parts) > 2:
                    abnormal_time = parts[0] + ":" + parts[1]

                print(f"  Analyzing data at: {abnormal_time}")

                # Step 1: Collect profiling data
                try:
                    profiling_data = collect_profiling_data(abnormal_time, ns, rca_data_path)
                except Exception as e:
                    logger.error("Failed to collect profiling data: %s", str(e))
                    print(f"  [ERROR] Failed to collect data: {e}")
                    rank_list.append(-1)
                    continue

                # Step 2: Build prompt
                if compare_normal and normal_profiling_data:
                    prompt = build_rca_prompt_with_normal(profiling_data, normal_profiling_data)
                else:
                    prompt = build_rca_prompt(profiling_data)

                logger.info("Prompt length: %d characters", len(prompt))
                print(f"  Prompt size: {len(prompt)} chars")

                # Step 3: Call LLM
                try:
                    llm_results = llm_root_cause_analysis(
                        prompt,
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                    )
                except Exception as e:
                    logger.error("LLM call failed: %s", str(e))
                    print(f"  [ERROR] LLM call failed: {e}")
                    rank_list.append(-1)
                    continue

                # Step 4: Evaluate
                rank = evaluate_single(llm_results, fault)
                rank_list.append(rank)

                # Print LLM's answer
                for r in llm_results[:3]:
                    idx = r.get("rank", "?")
                    print(f"  LLM Top-{idx}: service={r.get('service','?')}, "
                          f"type={r.get('fault_type','?')}")

                result_str = f"MATCH@{rank}" if rank > 0 else "MISS"
                print(f"  Result: {result_str}")

                # Save detailed result
                detailed_results.append({
                    "fault_number": fault_number,
                    "inject_time": inject_time,
                    "inject_pod": inject_pod,
                    "inject_type": inject_type,
                    "llm_results": llm_results,
                    "rank": rank,
                })

                # Rate limiting between API calls
                if fault_number < total_faults:
                    print(f"  Waiting {delay}s before next call...")
                    time.sleep(delay)

    # Final summary
    accuracy = compute_accuracy(rank_list, fault_number)
    print_results(accuracy, ns, mode="service")

    # Save detailed results to JSON
    output_file = dirname(__file__) + '/log/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + f'_llm_rca_results_{ns}_{provider}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "namespace": ns,
                "provider": provider,
                "model": model_name,
                "compare_normal": compare_normal,
                "timestamp": str(datetime.datetime.now()),
            },
            "accuracy": accuracy,
            "detailed_results": detailed_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Detailed results saved to: {output_file}")

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nezha LLM-based Root Cause Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_llm.py --ns hipster --provider gemini
  python main_llm.py --ns ts --provider openai --model gpt-4o
  python main_llm.py --ns hipster --provider gemini --compare-normal

Before running, set your API key:
  PowerShell:  $env:GEMINI_API_KEY="your_key_here"
  CMD:         set GEMINI_API_KEY=your_key_here
        """
    )

    parser.add_argument('--ns', default="hipster",
                        choices=["hipster", "ts"],
                        help='Namespace: hipster (OnlineBoutique) or ts (TrainTicket)')
    parser.add_argument('--provider', default="gemini",
                        choices=["gemini", "openai", "nvidia"],
                        help='LLM provider: gemini, openai, or nvidia')
    parser.add_argument('--model', default=None,
                        help='Model name (default: gemini-2.0-flash / gpt-4o / meta/llama-3.3-70b-instruct)')
    parser.add_argument('--api-key', default=None,
                        help='API key (overrides environment variable)')
    parser.add_argument('--delay', type=float, default=4.0,
                        help='Seconds to wait between API calls (default: 4)')
    parser.add_argument('--compare-normal', action='store_true',
                        help='Include normal baseline data in prompt for comparison')

    args = parser.parse_args()

    default_models = {"gemini": "gemini-2.0-flash", "openai": "gpt-4o", "nvidia": "meta/llama-3.3-70b-instruct"}
    model_disp = args.model or default_models.get(args.provider, "default")
    print(f"\n{'#'*60}")
    print(f"  Nezha LLM-based Root Cause Analysis")
    print(f"  Namespace:      {args.ns}")
    print(f"  LLM Provider:   {args.provider}")
    print(f"  Model:          {model_disp}")
    print(f"  Compare Normal: {args.compare_normal}")
    print(f"  API Call Delay: {args.delay}s")
    print(f"{'#'*60}\n")

    run_evaluation(
        ns=args.ns,
        provider=args.provider,
        model_name=args.model,
        compare_normal=args.compare_normal,
        api_key=args.api_key,
        delay=args.delay,
    )
