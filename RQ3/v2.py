"""
RQ4 消融實驗 (Ablation Experiment) — Ollama 本地部署版（支援中斷點續跑）
研究目的：評估「資料表示方式」與「Prompt 設計」對大語言模型 (LLM) 系統異常診斷品質的影響。

使用方式：
  pip install scikit-learn pandas requests
  python v2.py --csv data.csv

特性：
  支援自動斷點續跑。若按 Ctrl+C 中斷，直接重新執行同一行指令即可自動從上一筆紀錄繼續進行。

輸出檔案：
  rq4_results.csv     : 每筆測試資料的詳細預測紀錄（即時覆寫更新）
  rq4_summary.csv     : 6 組實驗組合的 F1-score / Top-1 / Top-3 彙總結果
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import f1_score

# ── 系統常數設定 ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
LABELS = ["Normal", "CPUOverload", "MemoryLeak",
          "LatencySpike", "NetworkDrop", "DiskFailure"]
LABEL_STR = ", ".join(LABELS)

# ── 自變項 A：資料表示方式 (Data Representation) ───────────────────────────────


def format_A1(row) -> str:
    """A1 模式：純數值（匿名特徵欄位，用以測試模型在無語義提示下的推理能力）"""
    return (
        f"feature_1 (cpu_%): {row['cpu_usage_percent']}\n"
        f"feature_2 (mem_mb): {row['memory_usage_mb']}\n"
        f"feature_3 (resp_ms): {row['response_time_ms']}\n"
        f"log_level: {row['log_level']}"
    )


def format_A2(row) -> str:
    """A2 模式：統計摘要（帶有嚴重程度標籤）"""
    cpu = float(row['cpu_usage_percent'])
    mem = float(row['memory_usage_mb'])
    rt = float(row['response_time_ms'])

    def lv(v, hi, mid):
        return "HIGH" if v > hi else ("MODERATE" if v > mid else "LOW")

    return (
        f"Service: {row['service_name']} | Log severity: {row['log_level']}\n"
        f"CPU usage is {lv(cpu, 80, 40)} at {cpu:.1f}%.\n"
        f"Memory usage is {lv(mem, 8000, 3000)} at {mem:.0f} MB.\n"
        f"Response time is {lv(rt, 3000, 1000)} at {rt:.0f} ms."
    )


def format_A3(row) -> str:
    """A3 模式：全自然語言敘述 (Natural Language Narrative)"""
    cpu = float(row['cpu_usage_percent'])
    mem = float(row['memory_usage_mb'])
    rt = float(row['response_time_ms'])

    cpu_d = "unusually high" if cpu > 80 else (
        "moderate" if cpu > 40 else "normal")
    mem_d = "unusually high" if mem > 8000 else (
        "elevated" if mem > 3000 else "normal")
    rt_d = "very slow" if rt > 3000 else (
        "above average" if rt > 1000 else "normal")

    return (
        f"At {row['timestamp']}, the {row['service_name']} service logged a {row['log_level']} event. "
        f"CPU usage appears {cpu_d}. "
        f"Memory consumption is {mem_d}. "
        f"Response time is {rt_d}."
    )


FORMAT_FNS = {"A1": format_A1, "A2": format_A2, "A3": format_A3}

# ── 自變項 B：Prompt 模組設計 (Prompt Engineering) ───────────────────────────


def build_B1(data_str: str) -> str:
    """B1 模式：Zero-shot 直接問答"""
    return (
        f"{data_str}\n\n"
        f"Based on the above metrics, what performance anomaly is occurring?\n"
        f"Reply with exactly one label from: {LABEL_STR}\n"
        f"Reply with the label only, nothing else."
    )


def build_B2(data_str: str) -> str:
    """B2 模式：Chain-of-Thought (CoT) 思維鏈提示"""
    return (
        f"{data_str}\n\n"
        f"Think step by step: which metric looks abnormal and why?\n"
        f"Then end your response with exactly: ANSWER: <label>\n"
        f"Choose the label from: {LABEL_STR}"
    )


PROMPT_FNS = {"B1": build_B1, "B2": build_B2}

# ── Ollama 本地 API 呼叫組件 ──────────────────────────────────────────────────


def call_ollama(prompt: str) -> str:
    """呼叫本地 Ollama 服務"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200}  # 固定隨機性並限制生成長度
    }
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            print(f"    [本地連線重試 第 {attempt+1} 次] 錯誤資訊: {e}")
            time.sleep(3)
    return "Unknown"

# ── 輸出解析與評估指標 ────────────────────────────────────────────────────────


def parse_pred(text: str) -> str:
    """解析 Ollama 的文本回應以擷取預測標籤"""
    # 優先匹配 "ANSWER: <label>"
    m = re.search(r'ANSWER:\s*(\w+)', text, re.IGNORECASE)
    if m:
        for label in LABELS:
            if label.lower() == m.group(1).lower():
                return label
    # 模糊關鍵字匹配
    for label in LABELS:
        if label.lower() in text.lower():
            return label
    return "Unknown"


def parse_top3(text: str) -> list:
    """提取回應中前 3 個不重複的可能標籤，用作 Top-3 評估"""
    found = []
    for label in LABELS:
        if label.lower() in text.lower() and label not in found:
            found.append(label)
        if len(found) == 3:
            break
    return found or ["Unknown"]


def top_k_acc(y_true, y_top3, k=1):
    """計算 Top-K 準確度"""
    return sum(1 for t, p in zip(y_true, y_top3) if t in p[:k]) / len(y_true)

# ── 主實驗控制核心 (含斷點管理) ───────────────────────────────────────────────


def run(df: pd.DataFrame, results_path: Path):
    combos = [(a, b) for a in ["A1", "A2", "A3"] for b in ["B1", "B2"]]

    # 斷點續跑機制：自動檢查已存在的進度檔案
    if results_path.exists():
        done_df = pd.read_csv(results_path)
        # 利用 (實驗組別, 資料索引) 作為唯一鍵值建立快取
        done_keys = set(zip(done_df["combo"], done_df["index"]))
        all_rows = done_df.to_dict("records")
        print(f"  ↻ 偵測到歷史紀錄：已完成 {len(done_keys)} 筆，將從上次中斷處繼續...")
    else:
        done_keys = set()
        all_rows = []

    for fmt, prm in combos:
        combo = f"{fmt}×{prm}"
        # 過濾掉該組別中已經跑完的資料索引
        remaining = [(i, row) for i, row in df.iterrows()
                     if (combo, i) not in done_keys]

        if not remaining:
            print(f"\n  ✓ 實驗組：{combo} 已全面完成，自動跳過。")
            continue

        print(f"\n{'─'*52}")
        print(f"  執行實驗組：{combo}  （剩餘份數：{len(remaining)} 筆）")
        print(f"{'─'*52}")

        for i, row in remaining:
            t0 = time.time()
            prompt = PROMPT_FNS[prm](FORMAT_FNS[fmt](row))
            output = call_ollama(prompt)
            pred = parse_pred(output)
            top3 = parse_top3(output)
            true = row["anomaly_type"]
            elapsed = time.time() - t0

            marker = "✓" if pred == true else "✗"
            print(
                f"  [{i+1:03d}] {marker} 真實標籤={true:<15} 模型預測={pred:<15} (耗時 {elapsed:.1f}秒)")

            all_rows.append({
                "combo": combo, "format": fmt, "prompt": prm,
                "index": i, "true_label": true,
                "predicted": pred, "top3": "|".join(top3),
                "raw_output": output[:150].replace("\n", " "),
            })

            # 每跑完一筆就立刻寫入檔案，確保意外中斷時資料不遺失
            pd.DataFrame(all_rows).to_csv(results_path, index=False)

    # 實驗完結：全數據分析與彙總
    df_res = pd.DataFrame(all_rows)
    summary = []
    print(f"\n{'='*52}")
    print(f"  最終實驗數據彙總報告")
    print(f"{'='*52}")
    for combo, grp in df_res.groupby("combo"):
        trues = grp["true_label"].tolist()
        preds = grp["predicted"].tolist()
        top3s = [r.split("|") for r in grp["top3"].tolist()]
        row_s = {
            "combo":    combo,
            "F1_macro": round(f1_score(trues, preds, average="macro", zero_division=0), 4),
            "Top1_Acc": round(top_k_acc(trues, top3s, k=1), 4),
            "Top3_Acc": round(top_k_acc(trues, top3s, k=3), 4),
            "n":        len(grp),
        }
        summary.append(row_s)
        print(
            f"  {combo:<12} F1={row_s['F1_macro']:.4f}  Top1={row_s['Top1_Acc']:.4f}  Top3={row_s['Top3_Acc']:.4f}")

    summary_path = results_path.parent / "rq4_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\n  詳細預測結果儲存於：{results_path}")
    print(f"  彙總報告結果儲存於：{summary_path}")

# ── 程式入口 (Main) ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True,
                        help="輸入的 CSV 資料集路徑 (預設應為 data.csv)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制測試筆數（0 代表執行全部資料，適合小樣本作測試）")
    args = parser.parse_args()

    # 讀取測試資料集
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(
        f"成功載入 {len(df)} 筆測試資料，各類別分佈: {df['anomaly_type'].value_counts().to_dict()}")

    run(df, Path(args.csv).parent / "rq4_results.csv")


if __name__ == "__main__":
    main()
