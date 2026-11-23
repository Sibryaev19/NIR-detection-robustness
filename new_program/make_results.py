#!/usr/bin/env python3
"""
make_results.py — расчёт метрик и построение графиков для отчёта.
Запуск: python make_results.py
Все настройки берутся из config.yaml.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# === Цветовая схема по типам искажений ===
DIST_COLORS = {
    "gaussian_noise": "#1f77b4",  # синий
    "jpeg_compress": "#ff7f0e",  # оранжевый
    "gaussian_blur": "#2ca02c",  # зелёный
    "brightness": "#d62728",  # красный
    "contrast": "#9467bd",  # фиолетовый
}
DIST_NAMES_RU = {
    "gaussian_noise": "Гауссовский шум",
    "jpeg_compress": "JPEG-сжатие",
    "gaussian_blur": "Гауссовское размытие",
    "brightness": "Изменение яркости",
    "contrast": "Изменение контраста",
    "clean": "Clean (baseline)",
}


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_level_id(level_id):
    """
    Разбирает level_id вида:
    - 'clean'
    - 'gaussian_noise_sigma_25'
    - 'brightness_alpha_1.0_beta_-40'
    Возвращает: (distortion_id, params_dict)
    """
    if level_id == "clean":
        return "clean", {}

    parts = level_id.split("_")
    known_types = {"gaussian_noise", "jpeg_compress", "gaussian_blur", "brightness", "contrast"}

    # Определяем тип искажения
    distortion_id = None
    for t in known_types:
        if level_id.startswith(t):
            distortion_id = t
            break

    if not distortion_id:
        # Fallback: если не нашли известный тип, считаем первым словом
        distortion_id = parts[0]

    # Парсим параметры из остатка строки
    param_str = level_id[len(distortion_id):].strip("_")
    params = {}
    if param_str:
        tokens = param_str.split("_")
        for i in range(0, len(tokens) - 1, 2):
            key = tokens[i]
            try:
                val = float(tokens[i + 1])
                if val == int(val):
                    val = int(val)
                params[key] = val
            except (ValueError, IndexError):
                continue

    return distortion_id, params


def get_primary_param(distortion_id, params):
    """Возвращает (param_name, param_value) для оси X"""
    mapping = {
        "gaussian_noise": "sigma",
        "jpeg_compress": "quality",
        "gaussian_blur": "sigma",
        "brightness": "beta",
        "contrast": "alpha",
    }
    param_name = mapping.get(distortion_id)
    if param_name and param_name in params:
        return param_name, params[param_name]
    # fallback
    if params:
        return next(iter(params.keys())), next(iter(params.values()))
    return None, None


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_metrics(gt_objects, pred_objects, iou_thresh=0.5):
    """Расчёт AP@50, Recall, F1, Mean IoU для одной серии"""
    if not gt_objects:
        return 0.0, 0.0, 0.0, 0.0

    sorted_preds = sorted(pred_objects, key=lambda x: x["confidence"], reverse=True)
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))
    gt_matched = [False] * len(gt_objects)
    matched_ious = []

    for i, pred in enumerate(sorted_preds):
        best_iou, best_gt = 0, -1
        for gt_idx, gt in enumerate(gt_objects):
            if gt_matched[gt_idx] or pred["class"] != gt["class"]:
                continue
            val = iou(pred["bbox"], gt["bbox"])
            if val > best_iou:
                best_iou, best_gt = val, gt_idx
        if best_iou >= iou_thresh and best_gt != -1:
            tp[i] = 1
            gt_matched[best_gt] = True
            matched_ious.append(best_iou)
        else:
            fp[i] = 1

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / len(gt_objects) if len(gt_objects) > 0 else np.zeros_like(cum_tp)
    precisions = np.divide(cum_tp, cum_tp + cum_fp, out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) != 0)

    # Интерполяция для AP
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = sum((recalls[n] - recalls[n - 1]) * precisions[n] for n in range(1, len(recalls)))

    final_r = recalls[-2]
    # Находим precision, соответствующий final recall
    idx = np.searchsorted(recalls, final_r, side="left")
    final_p = precisions[idx] if idx < len(precisions) else 0.0

    f1 = 2 * final_p * final_r / (final_p + final_r) if (final_p + final_r) > 0 else 0.0
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0

    return round(ap, 4), round(final_r, 4), round(f1, 4), round(mean_iou, 4)


def build_results_df(gt_data, pred_data, cfg):
    """Собирает DataFrame со всеми метриками"""
    allowed_classes = set(cfg["metrics"]["target_classes"])
    gt_dict = {item["filename"]: item["objects"] for item in gt_data}
    rows = []

    for level_id, exp in pred_data.items():
        distortion_id, params = parse_level_id(level_id)

        # Пропускаем неизвестные типы, если они не в нашей схеме цветов
        if distortion_id not in DIST_COLORS and distortion_id != "clean":
            continue

            # Собираем GT и предсказания по всем кадрам серии
        gt_all, pred_all = [], []
        for fr in exp["frames"]:
            if fr["filename"] not in gt_dict:
                continue
            gt_all += [o for o in gt_dict[fr["filename"]] if o["class"] in allowed_classes]
            pred_all += [p for p in fr["predictions"] if p["class"] in allowed_classes]

        ap, rec, f1, miou = calculate_metrics(gt_all, pred_all, cfg["metrics"]["iou_threshold"])
        param_name, param_value = get_primary_param(distortion_id, params)

        rows.append({
            "experiment_id": level_id,
            "distortion_id": distortion_id,
            "display_name": DIST_NAMES_RU.get(distortion_id, distortion_id),
            "is_clean": distortion_id == "clean",
            "primary_param_label": param_name,
            "primary_param_value": param_value,
            "AP@50": ap,
            "Recall": rec,
            "F1": f1,
            "Mean_IoU": miou,
        })

    df = pd.DataFrame(rows)

    # Вычисляем relative_AP
    clean_rows = df[df["is_clean"]]
    clean_ap = clean_rows["AP@50"].values[0] if not clean_rows.empty else None

    if clean_ap and clean_ap > 0:
        df["relative_AP"] = df.apply(
            lambda r: 1.0 if r["is_clean"] else round(r["AP@50"] / clean_ap, 4), axis=1
        )
    else:
        df["relative_AP"] = 0.0
        print("⚠️ Warning: Clean AP not found or zero. relative_AP set to 0.")

    return df, clean_ap


def save_tables(df, clean_ap, out_dir):
    """Сохраняет три итоговые CSV-таблицы"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Полная таблица
    df.to_csv(out_dir / "metrics_full.csv", index=False, encoding="utf-8-sig")

    # 2. Агрегация по типам искажений (исключая clean)
    distorted_df = df[~df["is_clean"]]
    if not distorted_df.empty:
        agg = distorted_df.groupby("distortion_id").agg(
            n_levels=("experiment_id", "count"),
            ap_mean=("AP@50", "mean"),
            ap_min=("AP@50", "min"),
            ap_max=("AP@50", "max"),
            rel_ap_mean=("relative_AP", "mean"),
            rel_ap_min=("relative_AP", "min"),
        ).reset_index()
        agg["quality_drop"] = agg["ap_max"] - agg["ap_min"]
        agg.to_csv(out_dir / "metrics_by_type.csv", index=False, encoding="utf-8-sig")

    # 3. Сводная таблица устойчивости
    mpc = distorted_df["AP@50"].mean() if not distorted_df.empty else 0
    rpc = mpc / clean_ap if clean_ap and clean_ap > 0 else 0

    robustness = pd.DataFrame([{
        "AP@50_clean": clean_ap,
        "mPC": round(mpc, 4),
        "rPC": round(rpc, 4),
        "total_series": len(df),
        "distorted_series": len(distorted_df),
    }])
    robustness.to_csv(out_dir / "metrics_robustness.csv", index=False, encoding="utf-8-sig")


def plot_summary_relative_ap(df, out_dir):
    """Главный сводный график: relative_AP по всем типам искажений"""
    plt.figure(figsize=(10, 6))

    plotted = False
    for dist_id, color in DIST_COLORS.items():
        sub = df[(df["distortion_id"] == dist_id) & (~df["is_clean"])].dropna(subset=["relative_AP"])
        if sub.empty:
            continue

        x = sub["primary_param_value"].values
        y = sub["relative_AP"].values

        # Сортируем по X для корректного соединения линий
        sorted_idx = np.argsort(x)
        plt.plot(x[sorted_idx], y[sorted_idx],
                 marker="o", label=DIST_NAMES_RU[dist_id], color=color, linewidth=2)
        plotted = True

    if plotted:
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (clean)")
        plt.xlabel("Значение параметра искажения")
        plt.ylabel("relative_AP (нормировано к clean)")
        plt.title("Устойчивость RT-DETR к различным типам искажений")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "relative_ap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_metric_by_type(df, metric, out_dir):
    """Строит 5 отдельных графиков для одной метрики по каждому типу искажения"""
    sub_dir = out_dir / metric.lower().replace("@", "").replace(" ", "_")
    sub_dir.mkdir(parents=True, exist_ok=True)

    clean_val = df[df["is_clean"]][metric].values
    has_clean = len(clean_val) > 0 and pd.notna(clean_val[0])

    for dist_id in DIST_COLORS.keys():
        sub = df[(df["distortion_id"] == dist_id) & (~df["is_clean"])].dropna(subset=[metric])
        if sub.empty:
            continue

        x = sub["primary_param_value"].values
        y = sub[metric].values

        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        plt.figure(figsize=(8, 5))
        plt.plot(x_sorted, y_sorted, marker="o", color=DIST_COLORS[dist_id], linewidth=2, label=DIST_NAMES_RU[dist_id])

        if has_clean:
            plt.axhline(clean_val[0], color="gray", linestyle="--", linewidth=1, label="Clean")

        param_label = sub["primary_param_label"].iloc[0]
        plt.xlabel(f"{param_label}" if param_label else "Уровень")
        plt.ylabel(metric)
        plt.title(f"{DIST_NAMES_RU[dist_id]}: {metric}")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(sub_dir / f"{dist_id}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_robustness_bar(df, out_dir):
    """Бар-чарт средней устойчивости по типам искажений"""
    distorted = df[~df["is_clean"]]
    if distorted.empty:
        return

    agg = distorted.groupby("distortion_id")["relative_AP"].mean().dropna()
    if agg.empty:
        return

    labels = [DIST_NAMES_RU.get(idx, idx) for idx in agg.index]
    colors = [DIST_COLORS.get(idx, "#777") for idx in agg.index]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, agg.values, color=colors, edgecolor="black")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (clean)")
    plt.ylabel("Средний relative_AP")
    plt.title("Средняя устойчивость по типам искажений")
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "relative_ap_mean_by_type.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # 1. Загрузка конфига
    cfg = load_config("config.yaml")

    # 2. Определение путей из конфига
    exp_name = cfg["experiment"]["name"]
    base_dir = Path(cfg["experiment"]["base_dir"])
    exp_dir = base_dir / exp_name
    metrics_dir = exp_dir / "metrics"

    gt_path = exp_dir / "ground_truth.json"
    pred_path = exp_dir / "predictions.json"

    print(f"📂 Эксперимент: {exp_name}")
    print(f"📁 Директория: {exp_dir}")

    if not gt_path.exists() or not pred_path.exists():
        print(f"❌ Ошибка: Отсутствуют файлы ground_truth.json или predictions.json в {exp_dir}")
        print("💡 Убедитесь, что вы запустили run_experiments.py перед этим скриптом.")
        return

    # 3. Загрузка данных
    print("📥 Загрузка данных...")
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    # 4. Расчёт метрик
    print("🧮 Расчёт метрик...")
    df, clean_ap = build_results_df(gt_data, pred_data, cfg)

    # 5. Сохранение таблиц
    print("💾 Сохранение таблиц...")
    save_tables(df, clean_ap, metrics_dir)

    # 6. Построение графиков
    print("📊 Построение графиков...")
    plot_summary_relative_ap(df, metrics_dir)

    metrics_to_plot = ["AP@50", "Recall", "F1", "Mean_IoU"]
    for metric in metrics_to_plot:
        plot_metric_by_type(df, metric, metrics_dir)

    plot_robustness_bar(df, metrics_dir)

    print(f"✅ Готово! Все результаты сохранены в: {metrics_dir}")


if __name__ == "__main__":
    main()
