"""
prompt_builder.py
Builds structured prompts for LLM-based root cause analysis.
"""


def _format_metrics(metric_summary):
    if not metric_summary:
        return "(No metric data available)"
    lines = []
    lines.append("| Service | Pod | CPU (%) | Memory (%) | Network P90 (ms) |")
    lines.append("|---------|-----|---------|------------|------------------|")
    for m in metric_summary:
        cpu = m.get("cpu_usage_percent", "N/A")
        mem = m.get("memory_usage_percent", "N/A")
        net = m.get("network_p90_ms", "N/A")
        lines.append(f"| {m['service']} | {m['pod']} | {cpu} | {mem} | {net} |")
    return "\n".join(lines)


def _format_alarms(alarm_summary):
    if not alarm_summary:
        return "(No resource alarms)"
    lines = []
    for a in alarm_summary:
        alarm_types = ", ".join(a["alarm_types"])
        lines.append(f"- **{a['service']}** (pod: {a['pod']}): {alarm_types} exceeded threshold")
    return "\n".join(lines)


def _format_traces(trace_summary):
    if not trace_summary:
        return "(No trace data available)"
    lines = []
    for i, chain in enumerate(trace_summary, 1):
        lines.append(f"Trace {i}: {chain}")
    return "\n".join(lines)


def _format_logs(log_summary):
    if not log_summary:
        return "(No log data available)"
    lines = []
    for service, messages in log_summary.items():
        lines.append(f"\n### {service}")
        for msg in messages:
            lines.append(f"  - {msg}")
    return "\n".join(lines)


def build_rca_prompt(profiling_data):
    """Build a complete RCA prompt for the LLM (fault-period only)."""
    services = profiling_data["services"]
    service_list_str = ", ".join(services)

    prompt = f"""You are an experienced Site Reliability Engineer (SRE). Below is profiling data collected from a microservice system during a fault period. Your task is to identify the root cause of the fault.

## System Architecture
This microservice system contains the following services: {service_list_str}

## Resource Metrics (at fault time: {profiling_data['time']})
{_format_metrics(profiling_data['metric_summary'])}

## Resource Alarms (threshold violations)
{_format_alarms(profiling_data['alarm_summary'])}

## Request Traces (sampled call chains)
{_format_traces(profiling_data['trace_summary'])}

## Log Summary (deduplicated key messages per service)
{_format_logs(profiling_data['log_summary'])}

## Task
Based on the above profiling data, identify which service (and what type of fault) is the most likely root cause of the system anomaly.

Fault types to consider:
- cpu_contention: CPU resource contention (high CPU usage)
- cpu_consumed: CPU over-consumption
- network_delay: Network latency issues
- memory: Memory pressure
- return: Incorrect return value / logic error in code
- exception: Application exception / error

Respond ONLY with a JSON object in this exact format (no markdown fences, no extra text):
{{
  "root_causes": [
    {{
      "rank": 1,
      "service": "service_name",
      "fault_type": "one of: cpu_contention, cpu_consumed, network_delay, memory, return, exception",
      "evidence": "brief explanation of why"
    }},
    {{
      "rank": 2,
      "service": "service_name",
      "fault_type": "fault_type",
      "evidence": "brief explanation"
    }},
    {{
      "rank": 3,
      "service": "service_name",
      "fault_type": "fault_type",
      "evidence": "brief explanation"
    }}
  ]
}}

Return exactly 3 root cause candidates ranked by likelihood. Respond with ONLY the JSON object."""

    return prompt


def build_rca_prompt_with_normal(profiling_data, normal_profiling_data):
    """Build a prompt that includes normal baseline for comparison (like Nezha's diff approach)."""
    services = profiling_data["services"]
    service_list_str = ", ".join(services)

    prompt = f"""You are an experienced Site Reliability Engineer (SRE). Below is profiling data from a microservice system. You are given BOTH the normal (healthy) baseline data and the fault-period data. Your task is to compare them and identify the root cause.

## System Architecture
This microservice system contains the following services: {service_list_str}

---
## NORMAL BASELINE (healthy system at {normal_profiling_data['time']})

### Normal Metrics
{_format_metrics(normal_profiling_data['metric_summary'])}

### Normal Alarms
{_format_alarms(normal_profiling_data['alarm_summary'])}

### Normal Traces
{_format_traces(normal_profiling_data['trace_summary'])}

### Normal Logs
{_format_logs(normal_profiling_data['log_summary'])}

---
## FAULT PERIOD (anomalous system at {profiling_data['time']})

### Fault Metrics
{_format_metrics(profiling_data['metric_summary'])}

### Fault Alarms
{_format_alarms(profiling_data['alarm_summary'])}

### Fault Traces
{_format_traces(profiling_data['trace_summary'])}

### Fault Logs
{_format_logs(profiling_data['log_summary'])}

---
## Task
Compare the normal baseline with the fault-period data. Identify which service experienced the most significant change and what type of fault is the most likely root cause.

Fault types to consider:
- cpu_contention: CPU resource contention (high CPU usage)
- cpu_consumed: CPU over-consumption
- network_delay: Network latency issues
- memory: Memory pressure
- return: Incorrect return value / logic error in code
- exception: Application exception / error

Respond ONLY with a JSON object in this exact format (no markdown fences, no extra text):
{{
  "root_causes": [
    {{
      "rank": 1,
      "service": "service_name",
      "fault_type": "one of: cpu_contention, cpu_consumed, network_delay, memory, return, exception",
      "evidence": "brief explanation comparing normal vs fault data"
    }},
    {{
      "rank": 2,
      "service": "service_name",
      "fault_type": "fault_type",
      "evidence": "brief explanation"
    }},
    {{
      "rank": 3,
      "service": "service_name",
      "fault_type": "fault_type",
      "evidence": "brief explanation"
    }}
  ]
}}

Return exactly 3 root cause candidates ranked by likelihood. Respond with ONLY the JSON object."""

    return prompt
