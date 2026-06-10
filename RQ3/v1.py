"""
RQ4 消融實驗 (Ablation Experiment)
研究目的：評估「資料表示方式」與「Prompt 設計」對大語言模型 (LLM) 系統異常診斷品質的影響。

使用方式：
  pip install groq scikit-learn pandas
  python v1.py --api_key gsk_YOUR_KEY --csv data.csv

輸出檔案：
  rq4_results.csv     : 每筆測試資料的詳細預測紀錄
  rq4_summary.csv     : 6 組實驗組合的 F1-score / Top-1 / Top-3 彙總結果
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from sklearn.metrics import classification_report, f1_score

# ── 系統常數設定 ─────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"
LABELS = ["Normal", "CPUOverload", "MemoryLeak",
          "LatencySpike", "NetworkDrop", "DiskFailure"]
LABEL_STR = ", ".join(LABELS)
RETRY_DELAY = 2   # 遇到 API Rate Limit 時的基礎等待秒數

# ── 自變項 A：資料表示方式 (Data Representation) ───────────────────────────────


def format_A1(row: dict) -> str:
    """A1 模式：純數值序列 (Key-Value)"""
    return (
        f"cpu_usage_percent: {row['cpu_usage_percent']}\n"
        f"memory_usage_mb: {row['memory_usage_mb']}\n"
        f"response_time_ms: {row['response_time_ms']}\n"
        f"log_level: {row['log_level']}\n"
        f"service: {row['service_name']}"
    )


def format_A2(row: dict) -> str:
    """A2 模式：統計摘要（帶有嚴重程度標籤）"""
    cpu = float(row['cpu_usage_percent'])
    mem = float(row['memory_usage_mb'])
    rt = float(row['response_time_ms'])

    # 依據門檻值進行文本標籤化
    cpu_label = "HIGH" if cpu > 80 else ("MODERATE" if cpu > 40 else "LOW")
    mem_label = "HIGH" if mem > 8000 else ("MODERATE" if mem > 3000 else "LOW")
    rt_label = "HIGH" if rt > 3000 else ("MODERATE" if rt > 1000 else "LOW")

    return (
        f"Service: {row['service_name']} | Log severity: {row['log_level']}\n"
        f"CPU usage is {cpu_label} at {cpu:.1f}%.\n"
        f"Memory usage is {mem_label} at {mem:.0f} MB.\n"
        f"Response time is {rt_label} at {rt:.0f} ms."
    )


def format_A3(row: dict) -> str:
    """A3 模式：全自然語言敘述 (Natural Language Narrative)"""
    cpu = float(row['cpu_usage_percent'])
    mem = float(row['memory_usage_mb'])
    rt = float(row['response_time_ms'])

    # 轉換為自然語言修飾詞
    cpu_desc = "unusually high" if cpu > 80 else (
        "moderate" if cpu > 40 else "normal")
    mem_desc = "unusually high" if mem > 8000 else (
        "elevated" if mem > 3000 else "normal")
    rt_desc = "very slow" if rt > 3000 else (
        "above average" if rt > 1000 else "normal")

    return (
        f"At {row['timestamp']}, the {row['service_name']} service logged a {row['log_level']} event. "
        f"CPU usage appears {cpu_desc}. "
        f"Memory consumption is {mem_desc}. "
        f"Response time is {rt_desc}."
    )


FORMAT_FNS = {"A1": format_A1, "A2": format_A2, "A3": format_A3}

# ── 自變項 B：Prompt 模組設計 (Prompt Engineering) ───────────────────────────


def build_prompt_B1(data_str: str) -> str:
    """B1 模式：Zero-shot 直接問答"""
    return (
        f"{data_str}\n\n"
        f"Based on the above metrics, what performance anomaly is occurring?\n"
        f"Answer with exactly one of: {LABEL_STR}\n"
        f"Reply with the label only, no explanation."
    )


def build_prompt_B2(data_str: str) -> str:
    """B2 模式：Chain-of-Thought (CoT) 思維鏈提示"""
    return (
        f"{data_str}\n\n"
        f"Think step by step about which metric looks abnormal, "
        f"then conclude with the anomaly type.\n"
        f"End your response with: ANSWER: <label>\n"
        f"Choose from: {LABEL_STR}"
    )


PROMPT_FNS = {"B1": build_prompt_B1, "B2": build_prompt_B2}

# ── 依變項解析：LLM 輸出提取 ─────────────────────────────────────────────────


def parse_prediction(text: str) -> str:
    """從 LLM 的文本回應中精準擷取預測的 Anomaly Label"""
    text = text.strip()

    # 優先解析 B2 CoT 格式：尋找 "ANSWER: <label>"
    m = re.search(r'ANSWER:\s*(\w+)', text, re.IGNORECASE)
    if m:
        candidate = m.group(1)
        for label in LABELS:
            if label.lower() == candidate.lower():
                return label

    # 若未偵測到標準結尾，則採用關鍵字模糊比對（不區分大小寫）
    for label in LABELS:
        if label.lower() in text.lower():
            return label

    return "Unknown"

# ── 評估指標計算 (Evaluation Metrics) ────────────────────────────────────────


def top_k_accuracy(y_true, y_pred_list, k=1):
    """計算 Top-K 準確率 (y_pred_list 為每筆資料預測標籤的候選清單)"""
    correct = sum(1 for true, preds in zip(
        y_true, y_pred_list) if true in preds[:k])
    return correct / len(y_true)


def parse_top3(text: str) -> list:
    """從 LLM 回應中，依序提取最多 3 個出現的候選標籤，用以評估 Top-3 準確率"""
    found = []
    for label in LABELS:
        if label.lower() in text.lower() and label not in found:
            found.append(label)
        if len(found) == 3:
            break
    if not found:
        found = ["Unknown"]
    return found

# ── 主實驗控制迴圈 ───────────────────────────────────────────────────────────


def run_experiment(df: pd.DataFrame, client: Groq, output_path: Path):
    results = []
    # 建立 3×2 = 6 組全面交織的實驗組合 (A1~A3 × B1~B2)
    combos = [(a, b) for a in ["A1", "A2", "A3"] for b in ["B1", "B2"]]

    for fmt_name, prompt_name in combos:
        combo = f"{fmt_name}×{prompt_name}"
        print(f"\n{'─'*50}")
        print(f"  執行實驗組：{combo}")
        print(f"{'─'*50}")

        preds, top3_preds, trues, raw_outputs = [], [], [], []

        for i, row in df.iterrows():
            data_str = FORMAT_FNS[fmt_name](row)
            prompt_str = PROMPT_FNS[prompt_name](data_str)

            # 呼叫 Groq API，並加入錯誤重試機制
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt_str}],
                        max_tokens=150,
                        temperature=0,  # 設為 0 以確保實驗具備可重複性 (Determinism)
                    )
                    output = resp.choices[0].message.content
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"    [重試中 第 {attempt+1} 次] 錯誤訊息: {e}")
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        output = "Unknown"

            # 預測結果解析
            pred = parse_prediction(output)
            top3 = parse_top3(output)
            true = row["anomaly_type"]

            preds.append(pred)
            top3_preds.append(top3)
            trues.append(true)
            raw_outputs.append(output[:120].replace("\n", " "))

            marker = "✓" if pred == true else "✗"
            print(f"  [{i+1:03d}] {marker} 真實標籤={true:<15} 模型預測={pred}")

            # 紀錄單筆詳細數據
            results.append({
                "combo": combo,
                "format": fmt_name,
                "prompt": prompt_name,
                "index": i,
                "true_label": true,
                "predicted": pred,
                "top3": "|".join(top3),
                "raw_output": raw_outputs[-1],
            })

            # 控制請求速率（配合 Groq 免費方案限制，約 30 req/min）
            time.sleep(0.5)

        # 計算該組實驗的評估指標
        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
        f1_report = classification_report(trues, preds, zero_division=0)
        top1 = top_k_accuracy(trues, top3_preds, k=1)
        top3_acc = top_k_accuracy(trues, top3_preds, k=3)

        print(f"\n  F1 (macro): {f1_macro:.4f}")
        print(f"  Top-1 Acc:  {top1:.4f}")
        print(f"  Top-3 Acc:  {top3_acc:.4f}")
        print(f"\n{f1_report}")

        # 即時寫入/更新每筆預測紀錄至 CSV
        records_path = output_path.parent / "rq4_results.csv"
        pd.DataFrame(results).to_csv(records_path, index=False)

    # 實驗結束：計算 6 組實驗的數據彙總
    df_results = pd.DataFrame(results)
    summary_rows = []
    for combo, grp in df_results.groupby("combo"):
        trues_g = grp["true_label"].tolist()
        preds_g = grp["predicted"].tolist()
        top3_g = [r.split("|") for r in grp["top3"].tolist()]
        summary_rows.append({
            "combo": combo,
            "F1_macro": round(f1_score(trues_g, preds_g, average="macro", zero_division=0), 4),
            "Top1_Acc": round(top_k_accuracy(trues_g, top3_g, k=1), 4),
            "Top3_Acc": round(top_k_accuracy(trues_g, top3_g, k=3), 4),
            "n": len(grp),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_path.parent / "rq4_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "="*50)
    print("  最終實驗數據彙總")
    print("="*50)
    print(df_summary.to_string(index=False))
    print(f"\n詳細紀錄已存至：{records_path}")
    print(f"彙總報告已存至：{summary_path}")

# ── 程式入口 (Main) ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Groq API 金鑰")
    parser.add_argument("--csv",     required=True,
                        help="輸入的 CSV 資料集路徑 (預設應為 data.csv)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="限制測試筆數（0 代表執行全部資料，適合小樣本作測試）")
    args = parser.parse_args()

    # 讀取測試資料集
    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)
    print(
        f"成功載入 {len(df)} 筆測試資料，各類別分佈: {df['anomaly_type'].value_counts().to_dict()}")

    client = Groq(api_key=args.api_key)
    run_experiment(df, client, Path(args.csv))


if __name__ == "__main__":
    main()
