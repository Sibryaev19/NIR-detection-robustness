import re
import cv2
from pathlib import Path
from ultralytics import RTDETR

def natural_sort_key(path: Path):
    """Сортирует имена файлов с учётом чисел (frame_2.jpg < frame_10.jpg)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem)]

def main():
    # 👉 Укажите путь к папке с кадрами
    input_dir = Path("source/video_1")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "detected_from_frames.mp4"

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {input_dir.resolve()}")

    # Собираем и сортируем изображения
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in image_exts],
        key=natural_sort_key
    )

    if not image_files:
        raise FileNotFoundError(f"В папке {input_dir} не найдено изображений с расширениями {image_exts}")

    print("⏳ Загрузка модели RT-DETR...")
    # device="0" если есть CUDA, иначе "cpu"
    model = RTDETR("rtdetr-l.pt")

    # Читаем первый кадр для получения разрешения
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Не удалось прочитать первый кадр: {image_files[0]}")

    height, width = first_frame.shape[:2]
    fps = 30.0  # 👉 Задайте желаемый FPS выходного видео (в папке кадров FPS нет)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("❌ Не удалось инициализировать VideoWriter. Проверьте кодек и права записи.")

    # Задержка показа для синхронизации с FPS (в миллисекундах)
    wait_delay = int(1000 / fps)

    print(f"🎬 Начало обработки: {width}x{height} @ {fps:.1f} FPS | Всего кадров: {len(image_files)}")
    processed = 0

    try:
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"⚠️ Пропуск повреждённого файла: {img_path.name}")
                continue

            # Детекция
            results = model.predict(
                source=frame,
                conf=0.35,
                imgsz=640,
                device="cpu",  # замените на "0" если есть NVIDIA GPU
                verbose=False,
                save=False,    # ❗ отключаем автосохранение Ultralytics
                show=False     # ❗ отключаем встроенное окно
            )

            # Рисуем bounding boxes
            annotated_frame = results[0].plot()

            # Записываем кадр
            writer.write(annotated_frame)
            processed += 1

            # Показываем превью
            cv2.imshow("RT-DETR video detection", annotated_frame)
            key = cv2.waitKey(wait_delay) & 0xFF
            if key == 27 or key == ord("q"):
                break

            if processed % 50 == 0 or processed == len(image_files):
                print(f"✅ Обработано: {processed}/{len(image_files)}")

    except KeyboardInterrupt:
        print("\n⏹️ Остановка пользователем")
    except Exception as e:
        print(f"\n⚠️ Ошибка: {e}")
    finally:
        writer.release()
        cv2.destroyAllWindows()

    print(f"\n🎉 Готово! Результат сохранён в: {output_video.resolve()}")
    print(f"📊 Всего кадров в видео: {processed}")


if __name__ == "__main__":
    main()