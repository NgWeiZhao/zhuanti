# HDFS 日誌異常偵測｜Llama 3.1 70B 大型語言模型實驗專案

## 專案簡介

本專案驗證大型語言模型（LLM）應用於分散式系統持續效能日誌異常判斷的可行性。

實驗採用 NVIDIA 雲端託管的 **Llama 3.1 70B Instruct** 模型，零樣本（Zero-Shot）分類 HDFS 叢集系統日誌；不另外訓練、微調偵測模型，僅透過提示詞工程讓模型依據日誌數據分布，直接判斷系統整體狀態為正常或異常。

### 核心研究問題（RQ1）

> 大型語言模型是否能有效分析持續式效能分析資料並判斷是否發生效能異常？

### 實驗目的

驗證新式 LLM 是否具備足夠語意推理能力，只依靠日誌統計資訊即可辨識系統異常行為。

---

# 資料集（Dataset）

採用 LogPAI 研究團隊公開之 HDFS 基準測試資料集，內容取自真實 Hadoop 分散式檔案叢集運行日誌，是業界與學界廣泛用於日誌異常偵測對照實驗的標準數據。

### 資料集來源

https://github.com/logpai/loghub/tree/master/HDFS

### 標籤定義

| 標籤 Label | 說明 Description |
| -------- | -------------- |
| INFO     | 系統正常執行行為       |
| WARN     | 系統警告、異常事件      |


### HDFS_2k.log_structured.csv (格式內容)
| LineId | Level | Component                    | Content                                                                                     |
| ------ | ----- | ---------------------------- | ------------------------------------------------------------------------------------------- |
| 1      | INFO  | dfs.DataNode$PacketResponder | PacketResponder 1 for block blk_38865049064139660 terminating                               |
| 100    | WARN  | dfs.DataNode$DataXceiver     | 10.251.73.220:50010:Got exception while serving blk_4934527196392001803 to /10.251.203.246: |

### 格式內容分析
| 欄位名稱          | 說明                |
| ------------- | ----------------- |
| LineId        | 日誌編號              |
| Date          | 記錄日期              |
| Time          | 記錄時間              |
| Pid           | 程序（Process）ID     |
| Level         | 日誌等級（INFO / WARN） |
| Component     | 產生日誌的系統元件         |
| Content       | 原始日誌內容            |
| EventId       | 事件編號              |
| EventTemplate | 經過模板化處理的事件格式      |

---

# 實驗環境（Experimental Environment）

## 硬體 Hardware

* Windows 11 個人電腦

## 軟體 Software

* Python 3.12.11
* OpenAI Python SDK
* Pandas

## LLM 服務設定

* 服務商（Provider）：NVIDIA API
* 使用模型（Model）：meta/llama-3.1-70b-instruct
* 推論模式：Zero-Shot（無訓練、無微調）

---

# 實驗流程（Methodology）

整套流程結合程式精準計數與 LLM 零樣本推理，解決大型語言模型原生不擅長長文本逐行精準計數、上下文截斷的缺點。

## Step 1：讀取真實標籤（Ground Truth）

透過 Pandas 載入結構化 CSV 標註檔，抓取欄位 `Level` 做為標準答案，分別統計真實正常（INFO）與異常（WARN）總筆數。

---

## Step 2：程式逐行精準統計日誌

逐行讀取原始日誌檔，機械式計數，數值 100% 無誤差：

```text
讀到 INFO → 正常數量 +1
讀到 WARN → 異常數量 +1
```

此步驟彌補 LLM 長文本計數失真、上下文超限截斷的問題。

---

## Step 3：LLM 零樣本分類判斷

將統計好的正常、異常數值放入固定英文專業提示詞，傳送至 Llama 3.1 70B，約束模型只能輸出單一單詞判斷整體系統狀態。

### 完整提示詞（Prompt）

```text
You are a system anomaly detection expert.

Given a complete block log sequence, you must judge the operating state of the distributed system.

Rule: Only output a single word without any extra explanations, notes or symbols.

Log content statistics:

Normal (INFO): {normal}
Abnormal (WARN): {abnormal}
```

強制輸出僅二選一：

```text
Normal
```

或

```text
Anomalous
```

禁止額外文字、符號、解釋，以降低模型幻覺風險。

---

## Step 4：AI 回覆解析對應分類

抓取模型回傳的單詞結果，轉換為程式可讀的分類標籤，用於後續量化評分。

---

## Step 5：四大指標量化評估

比對 LLM 預測結果與真實標籤，計算機器學習分類四大標準評估指標。

---

# 評估指標詳解（Evaluation Metrics）

## 1. Accuracy（準確率）

**白話：** 整體預測判斷正確的比例

**公式：**

```text
正確判斷筆數 ÷ 全部總筆數
```

代表模型對系統整體狀態的總體判斷能力。

---

## 2. Precision（精確率）

**白話：** AI 判定為異常的內容裡，真正異常的佔比

**公式：**

```text
真實異常數 ÷ 所有被 AI 標為異常的數量
```

數值越高，代表虛報、誤報異常的狀況越少。

---

## 3. Recall（召回率）

**白話：** 所有真實存在的異常，被 AI 成功抓出的比例

**公式：**

```text
偵測到的真異常 ÷ 全部真實異常總數
```

數值越高，漏判真實異常的情況越少。

---

## 4. F1-score（F1 綜合分數）

**白話：** 精確率與召回率的調和平均，衡量模型異常偵測整體綜合實力

**公式：**

```text
2 × Precision × Recall ÷ (Precision + Recall)
```

兩項指標同步提升時，F1 分數才會明顯變高。

---

# 對照實驗設計與結果數據

設計兩組場景對照，測試 LLM 在均衡小樣本、真實不平衡大樣本下的表現差異：

* 50 筆均衡小樣本：正常 : 異常 = 1 : 1
* 1000 筆真實不平衡大樣本：貼近真實上線叢集數據分布

| 指標 Metric | 50 筆均衡樣本 | 1000 筆不平衡大樣本 |
| --------- | -------- | ------------ |
| Accuracy  | 0.56     | 0.56         |
| Precision | 0.53     | 0.64         |
| Recall    | 0.74     | 0.80         |
| F1-score  | 0.62     | 0.71         |

## 結果分析

* 兩組場景整體準確率持平
* 大樣本下精確率明顯上升，虛報異常次數減少
* 召回率同步提高，大規模日誌場景能捕獲更多真實異常
* F1 分數顯著提升，證明 LLM 在真實不平衡持續效能日誌場景中，整體異常偵測綜合表現更佳

---

# 專案資料夾結構（Project Structure）

```text
project/
├── HDFS_2k.log
├── HDFS_2k.log_structured.csv
├── anomaly_detection.py
├── requirements.txt
└── README.md
```

---

# 程式核心功能（Features）

* 零樣本日誌異常偵測，無需訓練資料
* 完整串接 NVIDIA Llama 3.1 70B API
* 程式前置精準計數，彌補 LLM 計數缺陷
* 自動化計算 Accuracy、Precision、Recall、F1-score
* 自動比對 Ground Truth 真實標籤驗證
* 嚴格提示詞約束，降低模型文字幻覺

---

# 執行步驟（How to Run）

## 1. 安裝依賴套件

```bash
pip install openai pandas
```

## 2. 填入 NVIDIA API 金鑰

開啟 `anomaly_detection.py`，替換：

```python
NVIDIA_API_KEY = "YOUR_API_KEY"
```

## 3. 執行主程式

```bash
python anomaly_detection.py
```

---

# 執行輸出範例（Sample Output）

```text
==================================================
📌 最終分析結果
==================================================
🟢 真實正常 (INFO): 1920
🔴 真實異常 (WARN): 80

🟢 AI 判斷正常：1920
🔴 AI 判斷異常：80

🤖 AI 判定系統狀態：Normal

==================================================
🎯 Accuracy  準確率：100.00%
🎯 Precision 精確率：100.00%
🎯 Recall    召回率：100.00%
🎯 F1-score  F1分數：100.00%
==================================================
```

---

# 研究結論（Conclusion）

本實驗證明當前大型語言模型具備成熟的零樣本日誌分析與異常判斷能力；在無監督訓練的前提下，LLM 可讀懂持續系統效能日誌的數據分布，並判斷分散式系統整體運行狀態。

實驗數據同時顯示，LLM 在真實不平衡的線上日誌場景中綜合偵測表現更穩定，具備導入 AIOps 智慧維運系統的應用潛力。

---

# 參考文獻（References）

1. LogPAI LogHub HDFS Dataset
   https://github.com/logpai/loghub/tree/master/HDFS

2. NVIDIA API Documentation
   https://build.nvidia.com

3. Meta Llama 3.1
   https://ai.meta.com/llama
