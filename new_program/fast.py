#!/usr/bin/env python3
"""
combine_existing_plots.py — собирает готовые графики из папок в 2x2 композиции.
Запуск: python combine_existing_plots.py
Требует: matplotlib, Pillow (PIL)
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# === НАСТРОЙКИ ПУТЕЙ (поправь, если у тебя иначе) ===
BASE_DIR = "experiments/exp_merged_4videos/metrics"  # Папка с результатами

# Папки с готовыми графиками (относительно BASE_DIR)
METRIC_FOLDERS = {
    "AP@50": "ap50",
    "Recall": "recall",
    "F1": "f1",
    "Mean_IoU": "mean_iou"
}

# Типы искажений и их русские названия для заголовков
DISTORTIONS = {
    "gaussian_noise": "Гауссовский шум",
    "jpeg_compress": "JPEG-сжатие",
    "gaussian_blur": "Гауссовское размытие",
    "brightness": "Изменение яркости",
    "contrast": "Изменение контраста"
}

# Выходная папка для комбинированных графиков
OUTPUT_DIR = os.path.join(BASE_DIR, "combined")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def combine_2x2(images_dict, title, output_path, dpi=300):
    """
    Собирает 4 изображения в сетку 2×2 с общим заголовком.
    images_dict: dict с ключами ['AP@50', 'Recall', 'F1', 'Mean_IoU'] и путями к PNG
    """
    # Загружаем изображения
    imgs = {}
    for key, path in images_dict.items():
        if os.path.exists(path):
            imgs[key] = Image.open(path)
        else:
            print(f"⚠️ Не найден: {path}")
            return False

    # Создаём большой холст
    fig, axes = plt.subplots(2, 2, figsize=(24, 18), dpi=dpi)
    fig.suptitle(title, fontsize=28, fontweight="bold", y=0.98)

    # Порядок: верх-лево, верх-право, низ-лево, низ-право
    order = ["AP@50", "Recall", "F1", "Mean_IoU"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for metric, (row, col) in zip(order, positions):
        ax = axes[row, col]
        ax.imshow(imgs[metric])
        ax.set_title(metric, fontsize=20, pad=15)
        ax.axis('off')  # Убираем оси matplotlib, оставляем только картинку

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def main():
    print(f"📁 Ищем графики в: {BASE_DIR}")

    for dist_id, dist_name in DISTORTIONS.items():
        print(f"\n🔄 Обрабатываю: {dist_name}")

        # Собираем пути к 4 графикам для этого искажения
        img_paths = {}
        for metric, folder in METRIC_FOLDERS.items():
            # Ищем файл вида <dist_id>.png в папке метрики
            path = os.path.join(BASE_DIR, folder, f"{dist_id}.png")
            img_paths[metric] = path

        # Проверяем, что все 4 файла есть
        missing = [k for k, v in img_paths.items() if not os.path.exists(v)]
        if missing:
            print(f"  ⚠️ Пропущено: не найдены файлы для {missing}")
            continue

        # Собираем и сохраняем
        output_path = os.path.join(OUTPUT_DIR, f"{dist_id}_combined.png")
        if combine_2x2(img_paths, f"Влияние искажения: {dist_name}", output_path):
            print(f"  ✅ Сохранён: {output_path}")

    print(f"\n🎉 Готово! Комбинированные графики в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()