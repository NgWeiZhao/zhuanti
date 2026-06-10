"""
RQ4 消融實驗 (Ablation Experiment) — Ollama 本地部署版（修復時間負數問題 + 渲染圖檔 + 時間戳記）
研究目的：評估「資料表示方式」與「Prompt 設計」對大語言模型 (LLM) 系統異常診斷品質的影響。

使用方式：
  pip install scikit-learn pandas requests matplotlib seaborn
  python v3.py --csv data.csv

特性：
  1. 支援自動斷點續跑。
  2. 自動計算並輸出各實驗組別的文字版混淆矩陣。
  3. 自動將各實驗組別的混淆矩陣渲染成 PNG 圖檔。
  4. 輸出檔案名稱後面增加時間戳記。
  5. [Fix] 使用 perf_counter 保證耗時計算不為負數。
  6. 程式結束後自動保留完整 log 檔（與 CSV 同目錄，檔名含時間戳記）。
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Logger（全域，main() 初始化後即可使用）──────────────────────────────────────
logger = logging.getLogger("rq4")

# ── 系統常數設定 ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b"
LABELS = ["Normal", "CPUOverload", "MemoryLeak",
          "LatencySpike", "NetworkDrop", "DiskFailure"]
LABEL_STR = ", ".join(LABELS)
CM_LABELS = LABELS + ["Unknown"]

# ── 自變項 A：資料表示方式 (Data Representation) ───────────────────────────────

def _severity(value: float, hi: float, mid: float) -> str:
    """通用嚴重程度判斷（避免重複邏輯）"""
    return "HIGH" if value > hi else ("MODERATE" if value > mid else "LOW")


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
    rt  = float(row['response_time_ms'])
    return (
        f"Service: {row['service_name']} | Log severity: {row['log_level']}\n"
        f"CPU usage is {_severity(cpu, 80, 40)} at {cpu:.1f}%.\n"
        f"Memory usage is {_severity(mem, 8000, 3000)} at {mem:.0f} MB.\n"
        f"Response time is {_severity(rt, 3000, 1000)} at {rt:.0f} ms."
    )


def format_A3(row) -> str:
    """A3 模式：全自然語言敘述 (Natural Language Narrative)"""
    cpu = float(row['cpu_usage_percent'])
    mem = float(row['memory_usage_mb'])
    rt  = float(row['response_time_ms'])

    cpu_d = "unusually high" if cpu > 80 else ("moderate" if cpu > 40 else "normal")
    mem_d = "unusually high" if mem > 8000 else ("elevated" if mem > 3000 else "normal")
    rt_d  = "very slow"      if rt  > 3000 else ("above average" if rt > 1000 else "normal")

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
    """呼叫本地 Ollama 服務，失敗時最多重試 3 次"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            logger.warning("    [本地連線重試 第 %d 次] 錯誤資訊: %s", attempt + 1, e)
            time.sleep(3)
    return "Unknown"

# ── 輸出解析與評估指標 ────────────────────────────────────────────────────────


# 預先編譯正則表達式，避免每次呼叫重複編譯
_RE_ANSWER = re.compile(r'ANSWER:\s*(\w+)', re.IGNORECASE)
_LABELS_LOWER = {label.lower(): label for label in LABELS}


def parse_pred(text: str) -> str:
    """解析 Ollama 的文本回應以擷取預測標籤"""
    m = _RE_ANSWER.search(text)
    if m:
        candidate = _LABELS_LOWER.get(m.group(1).lower())
        if candidate:
            return candidate
    text_lower = text.lower()
    for lower, label in _LABELS_LOWER.items():
        if lower in text_lower:
            return label
    return "Unknown"


def parse_top3(text: str) -> list:
    """提取回應中前 3 個不重複的可能標籤，用作 Top-3 評估"""
    text_lower = text.lower()
    found = []
    for lower, label in _LABELS_LOWER.items():
        if lower in text_lower and label not in found:
            found.append(label)
        if len(found) == 3:
            break
    return found or ["Unknown"]


def top_k_acc(y_true: list, y_top3: list, k: int = 1) -> float:
    """計算 Top-K 準確度"""
    return sum(t in p[:k] for t, p in zip(y_true, y_top3)) / len(y_true)


def print_text_confusion_matrix(cm, labels: list) -> None:
    """在終端機漂亮地列印文字版混淆矩陣"""
    max_len = max(len(l) for l in labels)
    header = f"{' ' * max_len} | " + " ".join(f"{l[:6]:>6}" for l in labels)
    print("    " + header)
    print("    " + "-" * len(header))
    for label, row in zip(labels, cm):
        row_str = " ".join(f"{val:>6}" for val in row)
        print(f"    {label:<{max_len}} | {row_str}")


def _log_confusion_matrix(cm, labels: list) -> None:
    """透過 logger 輸出文字版混淆矩陣（同時寫入 log 檔）"""
    max_len = max(len(l) for l in labels)
    header = f"{' ' * max_len} | " + " ".join(f"{l[:6]:>6}" for l in labels)
    logger.info("    %s", header)
    logger.info("    %s", "-" * len(header))
    for label, row in zip(labels, cm):
        row_str = " ".join(f"{val:>6}" for val in row)
        logger.info("    %-*s | %s", max_len, label, row_str)


def plot_confusion_matrix(cm, labels: list, combo: str,
                          timestamp: str, output_dir: Path) -> None:
    """將混淆矩陣渲染成 PNG 圖檔"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix: {combo}')

    output_path = output_dir / f"{timestamp}_rq4_cm_{combo}.png"
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("  ✓ 混淆矩陣圖檔已儲存於：%s", output_path)

def plot_accuracy_comparison(summary: list, timestamp: str, output_dir: Path) -> None:
    """
    繪製六個實驗組別的正確率比較長條圖。
    X 軸：實驗組別（A1×B1 … A3×B2）
    Y 軸：正確率 %（Top-1 Accuracy × 100）
    每根長條頂端標示數值，方便直接比較。
    """
    combos   = [row["combo"]                        for row in summary]
    acc_pct  = [round(row["Top1_Acc"] * 100, 1)     for row in summary]

    # ── 配色：A1/A2/A3 各一色，B1/B2 以深淺區分 ──────────────────────────────
    palette = {
        "A1×B1": "#4C72B0", "A1×B2": "#7FA8D8",
        "A2×B1": "#DD8452", "A2×B2": "#F2B48A",
        "A3×B1": "#55A868", "A3×B2": "#90CFA0",
    }
    colors = [palette.get(c, "#999999") for c in combos]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(combos, acc_pct, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)

    # 長條頂端標數值
    for bar, val in zip(bars, acc_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, max(acc_pct) * 1.15 + 5)
    ax.set_xlabel("實驗組別 (資料表示 × Prompt 策略)", fontsize=12)
    ax.set_ylabel("Top-1 正確率 (%)", fontsize=12)
    ax.set_title("各實驗組別正確率比較", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(fig=fig, left=False, bottom=False)

    output_path = output_dir / f"{timestamp}_rq4_acc_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("  ✓ 正確率比較圖已儲存於：%s", output_path)


# ── 主實驗控制核心 (含斷點管理) ───────────────────────────────────────────────


def run(df: pd.DataFrame, results_path: Path) -> None:
    combos = [(a, b) for a in ("A1", "A2", "A3") for b in ("B1", "B2")]

    if results_path.exists():
        done_df   = pd.read_csv(results_path)
        done_keys = set(zip(done_df["combo"], done_df["index"]))
        all_rows  = done_df.to_dict("records")
        logger.info("  ↻ 偵測到歷史紀錄：已完成 %d 筆，將從上次中斷處繼續...", len(done_keys))
    else:
        done_keys = set()
        all_rows  = []

    for fmt, prm in combos:
        combo     = f"{fmt}×{prm}"
        remaining = [(i, row) for i, row in df.iterrows()
                     if (combo, i) not in done_keys]

        if not remaining:
            logger.info("\n  ✓ 實驗組：%s 已全面完成，自動跳過。", combo)
            continue

        logger.info("\n%s", "─" * 52)
        logger.info("  執行實驗組：%s  （剩餘份數：%d 筆）", combo, len(remaining))
        logger.info("%s", "─" * 52)

        format_fn = FORMAT_FNS[fmt]
        prompt_fn = PROMPT_FNS[prm]

        for i, row in remaining:
            t0 = time.perf_counter()

            prompt  = prompt_fn(format_fn(row))
            output  = call_ollama(prompt)
            elapsed = time.perf_counter() - t0

            pred = parse_pred(output)
            top3 = parse_top3(output)
            true = row["anomaly_type"]

            marker = "✓" if pred == true else "✗"
            logger.info("  [%03d] %s 真實標籤=%-15s 模型預測=%-15s (耗時 %.1f秒)",
                        i + 1, marker, true, pred, elapsed)

            all_rows.append({
                "combo":      combo,
                "format":     fmt,
                "prompt":     prm,
                "index":      i,
                "true_label": true,
                "predicted":  pred,
                "top3":       "|".join(top3),
                "raw_output": output[:150].replace("\n", " "),
            })

            pd.DataFrame(all_rows).to_csv(results_path, index=False)

    # ── 彙總報告 ──────────────────────────────────────────────────────────────
    df_res     = pd.DataFrame(all_rows)
    output_dir = results_path.parent
    timestamp  = results_path.stem.split('_')[0]
    summary    = []

    logger.info("\n%s", "=" * 52)
    logger.info("  最終實驗數據彙總報告")
    logger.info("%s", "=" * 52)

    for combo, grp in df_res.groupby("combo"):
        trues = grp["true_label"].tolist()
        preds = grp["predicted"].tolist()
        top3s = [r.split("|") for r in grp["top3"].tolist()]

        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
        t1_acc   = top_k_acc(trues, top3s, k=1)
        t3_acc   = top_k_acc(trues, top3s, k=3)

        logger.info("\n▶ 實驗組: %-12s F1=%.4f  Top1=%.4f  Top3=%.4f",
                    combo, f1_macro, t1_acc, t3_acc)

        cm = confusion_matrix(trues, preds, labels=CM_LABELS)
        logger.info("  [混淆矩陣 (縱軸為 True, 橫軸為 Pred)]")
        _log_confusion_matrix(cm, CM_LABELS)
        plot_confusion_matrix(cm, CM_LABELS, combo, timestamp, output_dir)

        row_s = {
            "combo":    combo,
            "F1_macro": round(f1_macro, 4),
            "Top1_Acc": round(t1_acc,   4),
            "Top3_Acc": round(t3_acc,   4),
            "n":        len(grp),
        }
        for r_idx, true_lbl in enumerate(CM_LABELS):
            for c_idx, pred_lbl in enumerate(CM_LABELS):
                row_s[f"cm_True_{true_lbl}_Pred_{pred_lbl}"] = int(cm[r_idx, c_idx])

        summary.append(row_s)

    summary_path = output_dir / f"{timestamp}_rq4_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    plot_accuracy_comparison(summary, timestamp, output_dir)
    logger.info("\n%s", "=" * 52)
    logger.info("  詳細預測結果儲存於：%s", results_path)
    logger.info("  彙總報告結果儲存於：%s (含完整混淆矩陣欄位)", summary_path)
    logger.info("  正確率比較圖儲存於：%s", output_dir / f"{timestamp}_rq4_acc_comparison.png")
    logger.info("  結束時間：%s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("%s", "=" * 52)

# ── Logging 初始化 ────────────────────────────────────────────────────────────


def setup_logging(log_path: Path) -> None:
    """
    設定 root logger：
      - StreamHandler  → 輸出至終端機（與原始 print 行為相同）
      - FileHandler    → 同步寫入 log 檔，程式結束後自動保留
    格式：僅訊息本身（不加時間戳記前綴），維持原始 log 外觀。
    """
    fmt = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("=" * 52)
    logger.info("  RQ4 消融實驗 執行紀錄")
    logger.info("  開始時間：%s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  推理後端：Ollama（本地部署，%s）", OLLAMA_URL)
    logger.info("  使用模型：%s", MODEL)
    logger.info("  ✎ Log 路徑：%s", log_path)


# ── 程式入口 (Main) ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True,
                        help="輸入的 CSV 資料集路徑 (預設應為 data.csv)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制測試筆數（0 代表執行全部資料，適合小樣本作測試）")
    args = parser.parse_args()

    # timestamp    = datetime.now().strftime("%m%d%H%M%S")
    timestamp    = Path(args.csv).name.split('_')[0] if '_' in args.csv else datetime.now().strftime("%m%d%H%M%S")
    results_path = Path(args.csv).parent / f"{timestamp}_rq4_results.csv"
    log_path     = Path(args.csv).parent / f"{timestamp}_rq4_log.log"
    setup_logging(log_path)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    dist = df['anomaly_type'].value_counts().to_dict()
    logger.info("  資料集檔案：%s", Path(args.csv).name)
    logger.info("  總筆數    ：%d", len(df))
    logger.info("  類別分佈  ：%s", dist)
    logger.info("=" * 52)

    run(df, results_path)


if __name__ == "__main__":
    main()