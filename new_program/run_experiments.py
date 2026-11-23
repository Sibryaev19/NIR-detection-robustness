#!/usr/bin/env python3
import cv2, json, yaml, random
import numpy as np
from pathlib import Path
from ultralytics import RTDETR
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DS_BASE = Path(CFG["dataset"]["base_dir"])
EXP_DIR = Path(CFG["experiment"]["base_dir"]) / CFG["experiment"]["name"]
PRED_PATH = EXP_DIR / "predictions.json"
GT_PATH_EXP = EXP_DIR / "ground_truth.json"
VIS_DIR = EXP_DIR / "visuals"
EXP_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def apply_degradation(img, deg_type, params):
    img = img.copy()
    if deg_type == "none": return img
    if deg_type == "gaussian_noise":
        sigma = params.get("sigma", 10)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if deg_type == "gaussian_blur":
        k = params.get("ksize", 5)
        sigma = params.get("sigma", 0)  # ← считываем sigma из конфига
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)  # ← передаём явно
    if deg_type == "jpeg_compress":
        q = params.get("quality", 50)
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if deg_type == "brightness_contrast":
        return cv2.convertScaleAbs(img, alpha=params.get("alpha", 1.0), beta=params.get("beta", 0))
    return img


def draw_predictions(img, preds, classes):
    out = img.copy()
    for p in preds:
        if p["class"] not in set(classes): continue
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{p['class']} {p['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def main():
    np.random.seed(42);
    random.seed(42)

    datasets = CFG["experiment"].get("datasets_to_merge", [])
    img_path_map, merged_gt_list = {}, []

    # 🔄 Логика объединения датасетов
    if datasets and all(isinstance(d, str) for d in datasets):
        print(f"🔗 Объединение датасетов: {datasets}")
        for ds_name in datasets:
            ds_dir = DS_BASE / ds_name
            gt_path = ds_dir / "ground_truth.json"
            if not gt_path.exists():
                print(f"⚠️ Пропущен {ds_name}: нет ground_truth.json");
                continue
            with open(gt_path) as f:
                gt_data = json.load(f)
            for item in gt_data:
                new_fname = f"{ds_name}_{item['filename']}"
                merged_gt_list.append({"filename": new_fname, "objects": item["objects"]})
                img_path_map[new_fname] = ds_dir / "frames" / item["filename"]
    else:
        # 📦 Режим одного датасета (как раньше)
        ds_name = CFG["dataset"]["name"]
        ds_dir = DS_BASE / ds_name
        gt_path = ds_dir / "ground_truth.json"
        if not gt_path.exists():
            print("❌ Сначала запустите make_annotation.py");
            return
        with open(gt_path) as f:
            gt_data = json.load(f)
        for item in gt_data:
            merged_gt_list.append(item)
            img_path_map[item["filename"]] = ds_dir / "frames" / item["filename"]

    # Сохраняем объединённую разметку в папку эксперимента
    with open(GT_PATH_EXP, "w") as f:
        json.dump(merged_gt_list, f, indent=2)
    print(f"✅ Разметка объединена и сохранена: {GT_PATH_EXP} ({len(img_path_map)} кадров)")

    all_preds = {}
    if PRED_PATH.exists():
        with open(PRED_PATH) as f: all_preds = json.load(f)
        print(f"📂 Загружены сохранённые предсказания ({len(all_preds)} серий)")

    print(f"⏳ Загрузка модели: {CFG['model']['path']}")
    model = RTDETR(CFG['model']['path'])
    allowed = set(CFG["metrics"]["target_classes"])

    for dist_cfg in tqdm(CFG["inference"]["distortions"], desc="Эксперименты"):
        for idx, lvl in enumerate(dist_cfg["levels"]):
            level_id = f"{dist_cfg['id']}_{'_'.join(f'{k}_{v}' for k, v in lvl.items())}" if lvl != {} else dist_cfg[
                "id"]
            if level_id in all_preds:
                print(f"⏭️ Пропущено: {level_id}");
                continue

            print(f"\n🔄 Запуск: {level_id}")
            frames_res = []
            vis_saved = 0

            for fname, fpath in tqdm(img_path_map.items(), desc="Кадр", leave=False):
                img = cv2.imread(str(fpath))
                if img is None:
                    frames_res.append({"filename": fname, "predictions": []});
                    continue

                proc_img = apply_degradation(img, dist_cfg["type"], lvl)
                res = model.predict(source=proc_img, conf=CFG["model"]["conf_threshold"],
                                    imgsz=CFG["model"]["imgsz"], device=CFG["model"]["device"],
                                    verbose=False, save=False, show=False)
                preds = []
                if res and res[0].boxes:
                    for b in res[0].boxes:
                        name = res[0].names.get(int(b.cls[0]), "").lower()
                        if name in allowed:
                            preds.append({"class": name, "bbox": b.xyxy[0].cpu().numpy().astype(int).tolist(),
                                          "confidence": float(b.conf[0])})
                frames_res.append({"filename": fname, "predictions": preds})

                if vis_saved < CFG["inference"]["visuals_per_level"]:
                    v_dir = VIS_DIR / level_id
                    v_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(v_dir / fname), draw_predictions(proc_img, preds, CFG["annotation"]["classes"]))
                    vis_saved += 1

            all_preds[level_id] = {"degradation": dist_cfg["type"], "params": lvl, "frames": frames_res}

    with open(PRED_PATH, "w") as f:
        json.dump(all_preds, f, indent=2)
    print(f"\n✅ Предсказания сохранены в {PRED_PATH}")


if __name__ == "__main__":
    main()






#!/usr/bin/env python3
# import cv2, json, yaml, random
# import numpy as np
# from pathlib import Path
# from ultralytics import RTDETR
# from tqdm import tqdm
# import warnings
#
# warnings.filterwarnings("ignore")
#
# with open("config.yaml", "r", encoding="utf-8") as f:
#     CFG = yaml.safe_load(f)
#
# DS_DIR = Path(CFG["dataset"]["base_dir"]) / CFG["dataset"]["name"]
# FRAMES_DIR = DS_DIR / "frames"
# GT_PATH = DS_DIR / "ground_truth.json"
#
# EXP_DIR = Path(CFG["experiment"]["base_dir"]) / CFG["experiment"]["name"]
# PRED_PATH = EXP_DIR / "predictions.json"
# VIS_DIR = EXP_DIR / "visuals"
# EXP_DIR.mkdir(parents=True, exist_ok=True)
# VIS_DIR.mkdir(parents=True, exist_ok=True)
#
#
# def apply_degradation(img, deg_type, params):
#     img = img.copy()
#     if deg_type == "none": return img
#     if deg_type == "gaussian_noise":
#         sigma = params.get("sigma", 10)
#         noise = np.random.normal(0, sigma, img.shape)
#         return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
#     if deg_type == "gaussian_blur":
#         k = params.get("ksize", 5)
#         if k % 2 == 0: k += 1
#         return cv2.GaussianBlur(img, (k, k), 0)
#     if deg_type == "jpeg_compress":
#         q = params.get("quality", 50)
#         _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
#         return cv2.imdecode(enc, cv2.IMREAD_COLOR)
#     if deg_type == "brightness_contrast":
#         return cv2.convertScaleAbs(img, alpha=params.get("alpha", 1.0), beta=params.get("beta", 0))
#     return img
#
#
# def draw_predictions(img, preds, classes):
#     out = img.copy()
#     for p in preds:
#         if p["class"] not in set(classes): continue
#         x1, y1, x2, y2 = map(int, p["bbox"])
#         cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(out, f"{p['class']} {p['confidence']:.2f}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return out
#
#
# def main():
#     np.random.seed(42);
#     random.seed(42)
#     if not GT_PATH.exists(): print("❌ Сначала запустите make_annotation.py"); return
#     with open(GT_PATH) as f:
#         gt_data = json.load(f)
#     img_paths = {item["filename"]: FRAMES_DIR / item["filename"] for item in gt_data}
#
#     all_preds = {}
#     if PRED_PATH.exists():
#         with open(PRED_PATH) as f: all_preds = json.load(f)
#         print(f"📂 Загружены сохранённые предсказания ({len(all_preds)} серий)")
#
#     print(f"⏳ Загрузка модели: {CFG['model']['path']}")
#     model = RTDETR(CFG['model']['path'])
#     allowed = set(CFG["metrics"]["target_classes"])
#
#     for dist_cfg in tqdm(CFG["inference"]["distortions"], desc="Эксперименты"):
#         for idx, lvl in enumerate(dist_cfg["levels"]):
#             level_id = f"{dist_cfg['id']}_{'_'.join(f'{k}_{v}' for k, v in lvl.items())}" if lvl != {} else dist_cfg[
#                 "id"]
#             if level_id in all_preds:
#                 print(f"⏭️ Пропущено: {level_id}");
#                 continue
#
#             print(f"\n🔄 Запуск: {level_id}")
#             frames_res = []
#             vis_saved = 0
#
#             for fname, fpath in tqdm(img_paths.items(), desc="Кадр", leave=False):
#                 img = cv2.imread(str(fpath))
#                 if img is None:
#                     frames_res.append({"filename": fname, "predictions": []});
#                     continue
#
#                 proc_img = apply_degradation(img, dist_cfg["type"], lvl)
#                 res = model.predict(source=proc_img, conf=CFG["model"]["conf_threshold"],
#                                     imgsz=CFG["model"]["imgsz"], device=CFG["model"]["device"],
#                                     verbose=False, save=False, show=False)
#                 preds = []
#                 if res and res[0].boxes:
#                     for b in res[0].boxes:
#                         name = res[0].names.get(int(b.cls[0]), "").lower()
#                         if name in allowed:
#                             preds.append({"class": name, "bbox": b.xyxy[0].cpu().numpy().astype(int).tolist(),
#                                           "confidence": float(b.conf[0])})
#                 frames_res.append({"filename": fname, "predictions": preds})
#
#                 if vis_saved < CFG["inference"]["visuals_per_level"]:
#                     v_dir = VIS_DIR / level_id
#                     v_dir.mkdir(parents=True, exist_ok=True)
#                     cv2.imwrite(str(v_dir / fname), draw_predictions(proc_img, preds, CFG["annotation"]["classes"]))
#                     vis_saved += 1
#
#             all_preds[level_id] = {"degradation": dist_cfg["type"], "params": lvl, "frames": frames_res}
#
#     with open(PRED_PATH, "w") as f:
#         json.dump(all_preds, f, indent=2)
#     print(f"\n✅ Предсказания сохранены в {PRED_PATH}")
#
#
# if __name__ == "__main__":
#     main()