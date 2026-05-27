"""
data_collector.py
Collects and summarizes profiling data (trace, log, metric) for a given time window.
Produces a compact summary suitable for feeding into an LLM prompt.
"""

import os
import re
import json
import logging
import datetime
import pandas as pd
import numpy as np

from os.path import dirname
from log import Logger
from alarm import get_metric_with_time, generate_alarm

log_path = dirname(__file__) + '/log/' + str(datetime.datetime.now().strftime(
    '%Y-%m-%d')) + '_nezha.log'
logger = Logger(log_path, logging.DEBUG, __name__).getlog()


def _extract_log_message(log_raw, pod):
    """Extract human-readable log message from the raw JSON log string."""
    try:
        if re.search(r'adservice', pod):
            return json.loads(log_raw)['log'].strip()
        elif re.search(r'cartservice', pod):
            return json.loads(log_raw)['log'].strip()
        elif re.search(r'checkoutservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'currencyservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'emailservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'frontend', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'paymentservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'productcatalogservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'recommendationservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'shippingservice', pod):
            return json.loads(json.loads(log_raw)['log'])['message'].strip()
        elif re.search(r'ts-', pod):
            return json.loads(log_raw)['log'].strip()
        else:
            return log_raw.strip()
    except Exception:
        return log_raw.strip() if isinstance(log_raw, str) else str(log_raw)


def _strip_dynamic_ids(message):
    """Remove TraceID/SpanID/UUIDs from log messages for deduplication."""
    msg = re.sub(r'TraceID:\s*[a-f0-9]+\s*', '', message)
    msg = re.sub(r'SpanID:\s*[a-f0-9]+\s*', '', msg)
    msg = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<UUID>', msg)
    msg = re.sub(r'userId=[^\s,]+', 'userId=<ID>', msg)
    msg = re.sub(r'product_ids=\[.*?\]', 'product_ids=[...]', msg)
    msg = re.sub(r'[A-Z0-9]{10}', '<ID>', msg)
    return msg.strip()


def _get_service_name(pod_name):
    """Extract service name from pod name."""
    service = pod_name.rsplit('-', 1)[0]
    service = service.rsplit('-', 1)[0]
    return service


def collect_metric_summary(time_str, data_path, ns):
    """Collect metric data for each pod at the given time."""
    metric_list = get_metric_with_time(time_str, data_path)
    summary = []
    for pod_metric in metric_list:
        pod = pod_metric["pod"]
        service = _get_service_name(pod)
        entry = {"pod": pod, "service": service}
        for m in pod_metric["metrics"]:
            if m["metric_type"] == "CpuUsageRate(%)":
                entry["cpu_usage_percent"] = round(m["metric_value"], 2)
            elif m["metric_type"] == "MemoryUsageRate(%)":
                entry["memory_usage_percent"] = round(m["metric_value"], 2)
            elif m["metric_type"] == "NetworkP90(ms)":
                entry["network_p90_ms"] = round(m["metric_value"], 2)
        summary.append(entry)
    return summary


def collect_alarm_summary(time_str, data_path, ns):
    """Generate resource alarms for the given time."""
    metric_list = get_metric_with_time(time_str, data_path)
    alarm_list = generate_alarm(metric_list, ns)
    summary = []
    for alarm in alarm_list:
        entry = {
            "pod": alarm["pod"],
            "service": _get_service_name(alarm["pod"]),
            "alarm_types": [a["metric_type"] for a in alarm["alarm"]]
        }
        summary.append(entry)
    return summary


def collect_trace_summary(time_str, data_path, max_traces=10):
    """Read trace data and produce a summary of call chains."""
    date = time_str.split(" ")[0]
    hour = time_str.split(" ")[1].split(":")[0]
    minute = time_str.split(" ")[1].split(":")[1]

    trace_file = data_path + "/" + date + "/trace/" + str(hour) + "_" + str(minute) + "_trace.csv"
    trace_id_file = data_path + "/" + date + "/traceid/" + str(hour) + "_" + str(minute) + "_traceid.csv"

    if not os.path.exists(trace_file) or not os.path.exists(trace_id_file):
        logger.warning("Trace file not found: %s", trace_file)
        return []

    trace_reader = pd.read_csv(
        trace_file, index_col='TraceID',
        usecols=['TraceID', 'SpanID', 'ParentID', 'PodName',
                 'OperationName', 'StartTimeUnixNano', 'EndTimeUnixNano', 'Duration'],
        engine='c'
    )
    trace_id_reader = pd.read_csv(trace_id_file, index_col=False, header=None, engine='c')

    all_trace_ids = trace_id_reader[0].tolist()
    sample_size = min(max_traces, len(all_trace_ids))
    indices = np.linspace(0, len(all_trace_ids) - 1, sample_size, dtype=int)
    sampled_ids = [all_trace_ids[i] for i in indices]

    trace_summaries = []
    for trace_id in sampled_ids:
        try:
            spans = trace_reader.loc[[trace_id], ['SpanID', 'ParentID', 'PodName',
                                                   'OperationName', 'Duration']]
            chain_parts = []
            for idx in range(len(spans)):
                pod = spans['PodName'].iloc[idx]
                op = spans['OperationName'].iloc[idx]
                duration = spans['Duration'].iloc[idx]
                service = _get_service_name(pod)
                chain_parts.append(f"{service}/{op}(duration={duration}us)")
            trace_summaries.append(" -> ".join(chain_parts))
        except Exception:
            continue

    return trace_summaries


def collect_log_summary(time_str, data_path, max_unique_per_pod=15):
    """Read log data and produce a deduplicated summary per service."""
    date = time_str.split(" ")[0]
    hour = time_str.split(" ")[1].split(":")[0]
    minute = time_str.split(" ")[1].split(":")[1]

    log_file = data_path + "/" + date + "/log/" + str(hour) + "_" + str(minute) + "_log.csv"

    if not os.path.exists(log_file):
        logger.warning("Log file not found: %s", log_file)
        return {}

    log_reader = pd.read_csv(log_file, index_col=False, usecols=['PodName', 'Log'], engine='c')

    service_logs = {}
    for idx in range(len(log_reader)):
        try:
            pod = log_reader['PodName'].iloc[idx]
            raw_log = log_reader['Log'].iloc[idx]
            if not isinstance(raw_log, str):
                continue

            service = _get_service_name(pod)
            message = _extract_log_message(raw_log, pod)
            dedup_key = _strip_dynamic_ids(message)

            if service not in service_logs:
                service_logs[service] = {"seen": set(), "messages": []}

            if dedup_key not in service_logs[service]["seen"]:
                service_logs[service]["seen"].add(dedup_key)
                service_logs[service]["messages"].append(dedup_key)
        except Exception:
            continue

    result = {}
    for service, data in service_logs.items():
        result[service] = data["messages"][:max_unique_per_pod]

    return result


def collect_profiling_data(abnormal_time, ns, rca_data_path):
    """
    Main entry: collect all profiling data for a given fault time.
    Returns dict with keys: metric_summary, alarm_summary, trace_summary, log_summary, services
    """
    logger.info("Collecting profiling data for %s (ns=%s)", abnormal_time, ns)

    metric_summary = collect_metric_summary(abnormal_time, rca_data_path, ns)
    alarm_summary = collect_alarm_summary(abnormal_time, rca_data_path, ns)
    trace_summary = collect_trace_summary(abnormal_time, rca_data_path, max_traces=8)
    log_summary = collect_log_summary(abnormal_time, rca_data_path, max_unique_per_pod=15)

    services = sorted(set(entry["service"] for entry in metric_summary))
    if not services:
        services = sorted(log_summary.keys())

    profiling_data = {
        "time": abnormal_time,
        "namespace": ns,
        "services": services,
        "metric_summary": metric_summary,
        "alarm_summary": alarm_summary,
        "trace_summary": trace_summary,
        "log_summary": log_summary,
    }

    logger.info("Collected: %d services, %d alarms, %d traces, %d log groups",
                len(services), len(alarm_summary), len(trace_summary), len(log_summary))

    return profiling_data
