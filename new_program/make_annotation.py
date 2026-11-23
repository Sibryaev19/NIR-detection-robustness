#!/usr/bin/env python3
import cv2, json, shutil, yaml
import numpy as np
from pathlib import Path
from ultralytics import RTDETR

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DS_DIR = Path(CFG["dataset"]["base_dir"]) / CFG["dataset"]["name"]
FRAMES_DIR = DS_DIR / "frames"
GT_PATH = DS_DIR / "ground_truth.json"

CLASSES = CFG["annotation"]["classes"]
CLASS_COLORS = {"car": (0, 255, 0), "truck": (255, 0, 0), "bus": (0, 165, 255)}
HOVER_COLOR = (255, 255, 0)
DRAW_COLOR = (255, 0, 255)

def get_images(d, step, count):
    p = Path(d)
    if not p.is_dir(): raise FileNotFoundError(f"❌ Нет папки: {d}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    return imgs[::step][:count] or []

class Annotator:
    def __init__(self, img_path, boxes, min_sz=20):
        self.img = cv2.imread(str(img_path))
        if self.img is None: raise ValueError(f"❌ Битый файл: {img_path}")
        self.img_path, self.min_sz = img_path, min_sz
        self.boxes = [b.copy() for b in boxes]
        self.hovered, self.drawing = None, False
        self.start, self.curr, self.has_moved = None, None, False
        self.next_cls = CLASSES[0]
        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", self._mouse)

    def _in_box(self, x, y, b):
        x1, y1, x2, y2 = map(int, b['bbox'])
        return x1 <= x <= x2 and y1 <= y <= y2

    def _update_hover(self, x, y):
        self.hovered = None
        for i in range(len(self.boxes) - 1, -1, -1):
            if self._in_box(x, y, self.boxes[i]):
                self.hovered = i; break

    def _mouse(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if self.hovered is not None:
                self.boxes.pop(self.hovered)
                self.hovered = None
            else:
                self.drawing = True
                self.start = self.curr = (x, y)
                self.has_moved = False
        elif e == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.curr = (x, y)
                if self.start != self.curr: self.has_moved = True
            else:
                self._update_hover(x, y)
        elif e == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.has_moved:
                x1, y1 = min(self.start[0], self.curr[0]), min(self.start[1], self.curr[1])
                x2, y2 = max(self.start[0], self.curr[0]), max(self.start[1], self.curr[1])
                if x2 - x1 >= self.min_sz and y2 - y1 >= self.min_sz:
                    self.boxes.append({"class": self.next_cls, "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                       "confidence": 1.0, "source": "manual"})
                self.drawing = False
                self.has_moved = False

    def _draw(self, img):
        out = img.copy()
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = map(int, b['bbox'])
            cls = b['class']
            col = HOVER_COLOR if i == self.hovered else (CLASS_COLORS.get(cls, (0,0,255)) if b.get('source')=='model' else (0,0,255))
            th = 4 if i == self.hovered else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            lbl = f"{cls} {b['confidence']:.2f}" if b.get('source')=='model' else cls
            cv2.putText(out, f"{lbl} ({b.get('source', 'manual')})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if self.drawing and self.start and self.curr:
            cv2.rectangle(out, self.start, self.curr, DRAW_COLOR, 2)
        h, w = out.shape[:2]
        cv2.putText(out, f"Файл: {self.img_path.name} | Боксов: {len(self.boxes)}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"След. класс: [{self.next_cls}] | [ЛКМ] Рис/Уд | [C] Смена | [1-3] Выбор | [SPACE] Далее | [Q] Выход",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return out

    def run(self):
        while True:
            if cv2.getWindowProperty("Annotator", cv2.WND_PROP_VISIBLE) < 1: return "exit"
            cv2.imshow("Annotator", self._draw(self.img))
            k = cv2.waitKey(10) & 0xFF
            if k == 32: return "next"
            if k in (ord('q'), 27): return "exit"
            if k == ord('c') and self.hovered is not None:
                idx = CLASSES.index(self.boxes[self.hovered]['class'])
                self.boxes[self.hovered]['class'] = CLASSES[(idx + 1) % len(CLASSES)]
            if k in (ord('1'), ord('2'), ord('3')):
                self.next_cls = CLASSES[k - ord('1')]
        return "next"

    def result(self):
        return {"filename": self.img_path.name, "width": self.img.shape[1],
                "height": self.img.shape[0], "objects": self.boxes}

    def close(self):
        cv2.destroyWindow("Annotator")

def predict_transport(model, path, conf):
    res = model.predict(str(path), conf=conf, verbose=False, save=False, show=False)
    if not res or not res[0].boxes: return []
    TARGETS = set(CLASSES)
    out = []
    for b in res[0].boxes:
        name = res[0].names.get(int(b.cls[0]), "").lower()
        if name in TARGETS:
            out.append({"class": name, "bbox": b.xyxy[0].cpu().numpy().astype(int).tolist(),
                        "confidence": float(b.conf[0]), "source": "model"})
    return out

def main():
    cfg_a = CFG["annotation"]
    src_imgs = get_images(CFG["dataset"]["source_frames_dir"], cfg_a["step"], cfg_a["num_images"])
    if not src_imgs: print("❌ Нет исходных изображений"); return

    # Создаём структуру датасета
    DS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📦 Копирование кадров в {FRAMES_DIR}...")
    for p in src_imgs:
        shutil.copy2(p, FRAMES_DIR / p.name)
    ds_imgs = sorted(list(FRAMES_DIR.glob("*.*")))

    print(f"⏳ Загрузка модели: {CFG['model']['path']}")
    model = RTDETR(CFG['model']['path'])
    annots = []
    print(f"📝 {len(ds_imgs)} кадров. Классы: {CLASSES}")

    for i, p in enumerate(ds_imgs, 1):
        print(f"\n[{i}/{len(ds_imgs)}] {p.name}")
        boxes = predict_transport(model, p, cfg_a["model_conf"])
        try:
            ann = Annotator(p, boxes, cfg_a["min_box_size"])
        except ValueError as e:
            print(f"  ⚠️ {e}"); continue
        st = ann.run()
        annots.append(ann.result())
        ann.close()
        print(f"  ✅ {len(ann.boxes)} объектов")
        if st == "exit": break

    with open(GT_PATH, "w") as f:
        json.dump(annots, f, indent=2)
    print(f"\n✅ Датасет сохранён в: {DS_DIR}")

if __name__ == "__main__":
    main()