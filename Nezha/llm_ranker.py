"""
llm_ranker.py
Calls LLM API (Google Gemini or OpenAI) to perform root cause analysis.
"""

import os
import re
import json
import time
import logging
import datetime

from os.path import dirname
from log import Logger

log_path = dirname(__file__) + '/log/' + str(datetime.datetime.now().strftime(
    '%Y-%m-%d')) + '_nezha.log'
logger = Logger(log_path, logging.DEBUG, __name__).getlog()


def _parse_json_response(text):
    """Parse JSON from LLM response, handles markdown fences and extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse JSON from LLM response: %s", text[:500])
    return None


def _extract_retry_delay(error_msg):
    """
    Extract the suggested retry delay from a 429 error message.
    e.g., 'Please retry in 36.198308212s.' -> 37 (rounded up + 1)
    Returns None if not found.
    """
    msg = str(error_msg)
    match = re.search(r'retry in (\d+(?:\.\d+)?)s', msg, re.IGNORECASE)
    if match:
        return int(float(match.group(1))) + 2  # Add 2s buffer
    match = re.search(r"'retryDelay':\s*'(\d+)s'", msg)
    if match:
        return int(match.group(1)) + 2
    return None


def _call_gemini(prompt, model_name, api_key, temperature=0.0):
    """Call Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature)
    )
    return response.text


def _call_openai(prompt, model_name, api_key, temperature=0.0):
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert SRE engineer. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_nvidia(prompt, model_name, api_key, temperature=0.0):
    """Call NVIDIA NIM API (OpenAI compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert SRE engineer. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def llm_root_cause_analysis(prompt, provider="gemini", model_name=None, api_key=None,
                            temperature=0.0, max_retries=5, retry_delay=5.0):
    """
    Call LLM API to perform root cause analysis.

    Returns:
        list of dicts: [{"rank": 1, "service": ..., "fault_type": ..., "evidence": ...}]
        Returns empty list on failure.
    """
    if provider == "gemini":
        # Use gemini-2.0-flash by default (free tier: 1500 req/day)
        # gemini-2.5-flash free tier is only 20 req/day — too low!
        model_name = model_name or "gemini-2.0-flash"
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        call_fn = _call_gemini
    elif provider == "openai":
        model_name = model_name or "gpt-4o"
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        call_fn = _call_openai
    elif provider == "nvidia":
        # Use meta/llama-3.3-70b-instruct as default NIM model
        model_name = model_name or "meta/llama-3.3-70b-instruct"
        api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        call_fn = _call_nvidia
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'openai', or 'nvidia'.")

    if not api_key:
        raise ValueError(
            f"No API key provided. Set {provider.upper()}_API_KEY environment variable "
            f"or pass --api-key parameter."
        )

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling %s (model=%s), attempt %d/%d",
                        provider, model_name, attempt, max_retries)

            raw_response = call_fn(prompt, model_name, api_key, temperature)
            logger.info("LLM raw response: %s", raw_response[:1000])

            parsed = _parse_json_response(raw_response)
            if parsed is None:
                logger.warning("Failed to parse on attempt %d", attempt)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                continue

            root_causes = parsed.get("root_causes", [])
            if not root_causes:
                logger.warning("No root_causes in parsed response")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                continue

            for rc in root_causes:
                rc["service"] = rc.get("service", "").strip().lower()
                rc["fault_type"] = rc.get("fault_type", "").strip().lower()

            logger.info("LLM returned %d root cause candidates", len(root_causes))
            return root_causes

        except Exception as e:
            error_str = str(e)
            logger.error("LLM call failed on attempt %d: %s", attempt, error_str)

            if attempt < max_retries:
                # Check if the error contains a suggested retry delay (429 rate limit)
                suggested_delay = _extract_retry_delay(error_str)
                if suggested_delay:
                    wait_time = suggested_delay
                    print(f"  [RATE LIMIT] API says retry in {wait_time}s, waiting...")
                    logger.info("Rate limited, waiting %d seconds as suggested by API", wait_time)
                else:
                    wait_time = retry_delay * attempt  # Exponential backoff

                time.sleep(wait_time)
            continue

    logger.error("All %d LLM call attempts failed", max_retries)
    return []
