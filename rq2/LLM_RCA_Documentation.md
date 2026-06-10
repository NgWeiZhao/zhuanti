# LLM-based Microservice Root Cause Analysis (LLM-RCA)

這份文件說明了在 Nezha 專案中所擴充的 **LLM-based RCA (基於大型語言模型的微服務根因分析)** 系統的實作細節與使用指南。

本模組旨在捨棄傳統複雜的圖挖掘演算法，改為利用大型語言模型（如 Llama 3, Gemini, OpenAI）強大的語意理解與推理能力，直接閱讀微服務的監控資料（Metrics, Logs, Traces），並以人類可讀的方式給出根因診斷報告。

---

## 1. 架構概述 (Architecture)

傳統 Nezha 依賴 `Drain3` 解析日誌並生成事件圖（Event Graph）來挖掘異常模式。
我們的 LLM-RCA 則採用了**摘要與提示詞工程 (Summarization & Prompt Engineering)** 的策略：

1. **資料收集與降噪 (Data Collection & Denoising)**：從原始 CSV 讀取資料，針對 TraceID/UUID 進行去重與壓縮，避免超過 LLM 的 Context Window 限制。
2. **提示詞構建 (Prompt Building)**：將過濾後的資源指標 (Metrics)、閾值告警 (Alarms)、調用鏈 (Traces) 和去重後的日誌 (Logs) 轉換為 Markdown 格式的綜合報告。
3. **LLM 推理 (LLM Reasoning)**：將提示詞發送至 LLM API (支援 NVIDIA NIM, Gemini, OpenAI)，讓 LLM 扮演 SRE (Site Reliability Engineer) 進行推理。
4. **結果解析與評估 (Evaluation)**：將 LLM 回傳的 JSON 結果提取出來，並與 Ground Truth 進行 Top-K 準確率對比。

---

## 2. 核心檔案清單 (Core Modules)

在專案根目錄下，我們新增了以下五個核心 Python 檔：

| 檔案名稱 | 核心功能說明 |
| :--- | :--- |
| `main_llm.py` | **主程式入口**。負責處理 CLI 參數、迴圈處理故障案例（Fault Cases）、排程整體 Pipeline、並將最終評估結果寫出為 JSON 檔。 |
| `data_collector.py` | **資料降噪與收集**。包含 `collect_metric_summary()`, `collect_log_summary()`, `collect_trace_summary()`。負責把龐大的 CSV 監控資料濃縮成有意義的子集（例如：日誌去除 UUID 後去重，Trace 均勻取樣 8 條）。 |
| `prompt_builder.py` | **提示詞工程**。將 `data_collector.py` 輸出的字典，排版成易讀的 Markdown 格式字串。支援兩種模式：單純丟入異常期資料，或加入正常基準線資料做差異比對（`compare_normal`）。 |
| `llm_ranker.py` | **模型介面與 API 呼叫**。封裝了調用 Google GenAI、OpenAI SDK 以及 NVIDIA NIM 的邏輯。內建指數退避重試機制 (Exponential Backoff) 以應對 429 Rate Limit。 |
| `evaluator.py` | **自動化評估**。負責比對 LLM 輸出的預測微服務與實際 Ground Truth，計算 AS@1 (Top-1 準確率) 與 AS@3 (Top-3 準確率)。 |

---

## 3. 環境設定與需求 (Setup & Prerequisites)

執行 LLM-RCA 之前，請確保安裝以下 Python 套件（除了原本 Nezha 的依賴外）：

```bash
pip install openai google-genai pandas
```

**設定 API Keys**
本系統支援多個 LLM 提供商。執行前必須在終端機設定環境變數：

**Windows (PowerShell):**
```powershell
# 如果你要用 NVIDIA NIM (例如 Llama 3)
$env:NVIDIA_API_KEY="nvapi-xxxxxxx"

# 如果你要用 Google Gemini
$env:GEMINI_API_KEY="AIzaSy-xxxxxxx"

# 如果你要用 OpenAI (GPT-4)
$env:OPENAI_API_KEY="sk-xxxxxxx"
```

---

## 4. 執行指南 (Usage)

系統統一由 `main_llm.py` 執行。以下是幾個常見的執行情境：

### 4.1 基本執行：使用 NVIDIA NIM 平台 (推薦)
預設會呼叫 NVIDIA NIM 上的 `meta/llama-3.3-70b-instruct` 模型，對 `hipster` (OnlineBoutique) 數據集進行分析。

```bash
python main_llm.py --ns hipster --provider nvidia
```

### 4.2 啟用「正常基準線對比」(Compare Normal - 推薦開啟)
LLM 若只看故障資料，容易被背景的「常態性告警」（例如 Adservice 記憶體長期處於 80%）誤導。加上 `--compare-normal` 參數後，系統會在 Prompt 中附上系統正常時的數據，讓 LLM 進行前後對比差異，能大幅提升準確率（實驗顯示從 55% 提升至 73%）。

```bash
python main_llm.py --ns hipster --provider nvidia --compare-normal
```

### 4.3 切換不同模型
透過 `--model` 可以指定特定提供商的其他模型。例如使用 Minimax 模型：

```bash
python main_llm.py --ns hipster --provider nvidia --model minimaxai/minimax-m2.7
```

### 4.4 參數說明總表

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--ns` | `hipster` | 目標系統空間：`hipster` (OnlineBoutique) 或 `ts` (TrainTicket) |
| `--provider` | `gemini` | API 提供商，支援 `nvidia`, `gemini`, `openai` |
| `--model` | (依 provider 定) | 自訂模型名稱。若留空則使用該 provider 的預設模型 |
| `--api-key` | `None` | 直接傳入 API Key（不建議，優先讀取環境變數） |
| `--delay` | `4.0` | 兩次 LLM 請求間的暫停秒數，避免觸發 429 Rate Limit |
| `--compare-normal`| `False` | 是否在提示詞中包含「正常時期 (Baseline)」的資料供 LLM 差異對比 |

---

## 5. 輸出與日誌結構 (Outputs)

執行完畢後，系統會在終端機印出整體的準確率報告：
```text
============================================================
  LLM RCA Results (hipster, service level)
============================================================
  Total faults:   56
  Not found:      8
------------------------------------------------------------
  Top-1 Accuracy: 73.21% (41/56)
  Top-3 Accuracy: 85.71% (48/56)
============================================================
```

### 詳細 JSON 報告
系統會將每一次故障診斷的詳細過程（包括 LLM 的自然語言解釋）存放在 `log/YYYY-MM-DD_llm_rca_results_{ns}_{provider}.json`。

**JSON 節錄範例：**
```json
{
  "fault_number": 5,
  "inject_pod": "cartservice-579f59597d-wc2lz",
  "inject_type": "network_delay",
  "llm_results": [
    {
      "rank": 1,
      "service": "cartservice",
      "fault_type": "network_delay",
      "evidence": "cartservice NetworkP90=252.6ms exceeds threshold, and GetCart operations consistently show 240-260ms latency across all 8 traces. This network delay is the dominant factor causing overall request latency."
    }
  ],
  "rank": 1
}
```
*這份 JSON 檔極具價值，其中的 `evidence` 欄位展現了 LLM 基於什麼數據推斷出根因，這解決了傳統方法缺乏可解釋性的痛點。*
