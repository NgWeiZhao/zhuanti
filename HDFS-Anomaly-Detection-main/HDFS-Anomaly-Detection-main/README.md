# HDFS Log Anomaly Detection using Llama 3.1 70B

## Project Overview

This project explores the feasibility of applying Large Language Models (LLMs) to distributed system log anomaly detection.

The experiment utilizes NVIDIA-hosted Llama 3.1 70B Instruct to perform zero-shot classification on HDFS system logs. Instead of training a dedicated anomaly detection model, the LLM is directly prompted to determine whether the system state is Normal or Anomalous.

The objective is to investigate whether modern LLMs possess sufficient reasoning capabilities to identify abnormal system behavior from log statistics alone.

---

## Dataset

This project uses the HDFS (Hadoop Distributed File System) log dataset provided by the LogPAI research team.

Dataset Source:

https://github.com/logpai/loghub/tree/master/HDFS

### Dataset Description

The dataset contains real-world HDFS system logs collected from Hadoop clusters.

Each log entry is labeled according to its operating status:

| Label | Description                |
| ----- | -------------------------- |
| INFO  | Normal system behavior     |
| WARN  | Abnormal or warning events |

The dataset is widely used as a benchmark for log anomaly detection research.

---

## Experimental Environment

### Hardware

* Windows 11 PC

### Software

* Python 3.x
* OpenAI Python SDK
* Pandas

### LLM Service

Provider:

* NVIDIA API

Model:

* meta/llama-3.1-70b-instruct

---

## Methodology

The experiment follows the workflow below:

### Step 1: Load Ground Truth Labels

The structured CSV file is loaded using Pandas.

The Level column is used as the ground truth label source.

Example:

* INFO → Normal
* WARN → Abnormal

The total number of normal and abnormal logs is calculated.

---

### Step 2: Program-Based Log Counting

The raw log file is parsed line by line.

Rules:

* INFO → Normal Count +1
* WARN → Abnormal Count +1

This stage provides precise statistical information about the log sequence.

---

### Step 3: Zero-Shot LLM Classification

The calculated log statistics are injected into a prompt and sent to Llama 3.1 70B.

Prompt:

```text
You are a system anomaly detection expert.

Given a complete block log sequence, you must judge the operating state of the distributed system.

Rule: Only output a single word without any extra explanations, notes or symbols.

Optional output word:
Normal / Anomalous
```

The model receives:

```text
Normal (INFO): X
Abnormal (WARN): Y
```

and returns:

```text
Normal
```

or

```text
Anomalous
```

---

### Step 4: Prediction Parsing

The LLM output is parsed and converted into a machine-readable prediction result.

---

### Step 5: Performance Evaluation

The prediction result is compared with the ground-truth labels.

The following evaluation metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-score

---

## Evaluation Metrics

### Accuracy

Measures the overall correctness of predictions.

### Precision

Measures how many predicted anomalies are truly anomalous.

### Recall

Measures how many true anomalies are successfully detected.

### F1-score

The harmonic mean of Precision and Recall.

---

## Project Structure

```text
project/
│
├── HDFS_2k.log
├── HDFS_2k.log_structured.csv
│
├── anomaly_detection.py
│
├── README.md
│
└── requirements.txt
```

---

## Program Features

* Zero-Shot Anomaly Detection
* NVIDIA API Integration
* Llama 3.1 70B Evaluation
* Automated Metric Calculation
* Structured Log Analysis
* Ground Truth Verification

---

## How to Run

### Install Dependencies

```bash
pip install openai pandas
```

### Configure NVIDIA API Key

Replace:

```python
NVIDIA_API_KEY = "YOUR_API_KEY"
```

with your own NVIDIA API Key.

### Execute

```bash
python anomaly_detection.py
```

---

## Example Output

```text
📌 Final Analysis Result

True Normal (INFO): 1850
True Abnormal (WARN): 150

AI Predicted Normal: 1850
AI Predicted Abnormal: 150

AI Decision: Anomalous

Accuracy : 100.00%
Precision: 100.00%
Recall   : 100.00%
F1-score : 100.00%
```

---

## Research Objective

This project investigates whether Large Language Models can be utilized as anomaly detectors without additional training.

The study focuses on:

* Log understanding capability
* Zero-shot reasoning ability
* System anomaly identification performance
* LLM applicability in AIOps scenarios

---

## References

1. LogPAI LogHub Dataset

https://github.com/logpai/loghub/tree/master/HDFS

2. NVIDIA API Documentation

https://build.nvidia.com

3. Meta Llama 3.1

https://ai.meta.com/llama
