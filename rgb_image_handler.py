"""Wound detector for RGB foot images on Raspberry Pi (YOLO exported to LiteRT / .tflite).

    from wound_predictor import WoundPredictor
    model = WoundPredictor("wound.tflite", class_names=["wound"])
    confidence, label = model.predict(left_rgb, right_rgb)          # both feet
    confidence, label = model.predict_single(foot_rgb)              # one image
    confidence, label, info = model.predict(left_rgb, right_rgb, return_details=True)
    model.predict(left_rgb, right_rgb, save_path="results")   # -> results/wound_20261121_234700.jpg
"""
import os
from datetime import datetime
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

try:
    from PIL import Image, ImageDraw, ImageFont             # resize like YOLO training + annotated images
except ImportError:
    Image = ImageDraw = ImageFont = None


def _output_path(save_path, stem, default_ext, allowed_exts):
    """Folder -> folder/<stem>_YYYYmmdd_HHMMSS<ext> (never overwrites). A full file path is used as is."""
    if os.path.splitext(save_path)[1].lower() in allowed_exts:
        path = save_path
    else:
        name = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
        path = os.path.join(save_path, name + default_ext)
        i = 1
        while os.path.exists(path):                       # two saves in the same second
            path = os.path.join(save_path, f"{name}_{i}{default_ext}"); i += 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


class WoundPredictor:
    LABEL_POSITIVE = "Wound Detected"
    LABEL_NEGATIVE = "No Wound"

    def __init__(self, model_path, class_names=None, conf_threshold=0.25, iou_threshold=0.45,
                 min_box_frac=0.0005, input_is_bgr=False, num_threads=4):
        """
        model_path     : path to the YOLO .tflite file (Ultralytics export, format="litert")
        class_names    : list of class names in training order, e.g. ["wound"]
        conf_threshold : minimum box score to count as a wound (raise it if you get false alarms)
        iou_threshold  : overlap above which duplicate boxes are merged
        min_box_frac   : ignore boxes smaller than this fraction of the image area (noise)
        input_is_bgr   : True if you pass OpenCV images (BGR); they are converted to RGB
        num_threads    : CPU threads for LiteRT (Pi 3 has 4 cores)
        """
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_box_frac = min_box_frac
        self.input_is_bgr = input_is_bgr

        self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shp = self.inp["shape"]
        self.nchw = int(shp[1]) == 3                         # LiteRT exports may be channels-first
        self.size = int(shp[2]) if self.nchw else int(shp[1])

    # ------------------------------------------------------------------ public API
    def predict(self, left, right, save_path=None, return_details=False, file_prefix=None):
        """left, right: HxWx3 uint8 images of each foot.
        save_path: folder to save the annotated image in (both feet side by side, boxes + result text;
                   file name generated with a timestamp, e.g. wound_20261121_234700.jpg). None = don't save.
        file_prefix: optional start of the file name, e.g. "patient01" -> patient01_wound_<time>.jpg
        Returns (confidence, label). Wound found -> confidence = best wound score;
        no wound -> confidence = 1 - best (rejected) score."""
        res_l, res_r = self._run(left), self._run(right)
        boxes = res_l["boxes"] + res_r["boxes"]
        best = max(res_l["best_score"], res_r["best_score"])
        confidence, label = self._decide(boxes, best)
        info = {"left": res_l["boxes"], "right": res_r["boxes"],
                "left_wound": bool(res_l["boxes"]), "right_wound": bool(res_r["boxes"]),
                "best_score": best, "conf_threshold": self.conf_threshold, "saved_to": None}
        if save_path:
            panels = [(self._prep(left), res_l["boxes"], "LEFT FOOT"),
                      (self._prep(right), res_r["boxes"], "RIGHT FOOT")]
            info["saved_to"] = self._save_annotated(panels, confidence, label, save_path, file_prefix)
        if return_details:
            return confidence, label, info
        return confidence, label

    def predict_single(self, image, save_path=None, return_details=False, file_prefix=None):
        res = self._run(image)
        confidence, label = self._decide(res["boxes"], res["best_score"])
        info = {"boxes": res["boxes"], "best_score": res["best_score"], "saved_to": None}
        if save_path:
            info["saved_to"] = self._save_annotated([(self._prep(image), res["boxes"], "FOOT")],
                                                    confidence, label, save_path, file_prefix)
        if return_details:
            return confidence, label, info
        return confidence, label

    def detect(self, image):
        """Just the boxes: [{'box': (x1,y1,x2,y2), 'conf': float, 'label': str}, ...] in original pixels."""
        return self._run(image)["boxes"]

    @staticmethod
    def draw(image, boxes, color=(255, 0, 0), thickness=3):
        """Return a copy of the image with boxes drawn (numpy only, no OpenCV needed)."""
        img = np.ascontiguousarray(image).copy()
        h, w = img.shape[:2]
        for d in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
            t = thickness
            img[y1:y1 + t, x1:x2 + 1] = color; img[max(y2 - t + 1, 0):y2 + 1, x1:x2 + 1] = color
            img[y1:y2 + 1, x1:x1 + t] = color; img[y1:y2 + 1, max(x2 - t + 1, 0):x2 + 1] = color
        return img

    # ------------------------------------------------------------------ inference
    def _decide(self, boxes, best_score):
        if boxes:
            return max(b["conf"] for b in boxes), self.LABEL_POSITIVE
        return 1.0 - best_score, self.LABEL_NEGATIVE

    def _prep(self, image):
        """Any input image -> HxWx3 uint8 RGB."""
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        if self.input_is_bgr:
            img = img[..., ::-1]
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def _run(self, image):
        img = self._prep(image)

        x, scale, (px, py) = self._letterbox(img)
        x = x.astype(np.float32) / 255.0
        if self.nchw:
            x = x.transpose(2, 0, 1)
        if self.inp["dtype"] in (np.uint8, np.int8):         # fully-quantised model
            s, z = self.inp["quantization"]
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / s + z), info.min, info.max).astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0]
        if y.dtype != np.float32:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * (s if s else 1.0)

        boxes, scores, classes, best = self._decode(y.astype(np.float32))
        if boxes.size and boxes.max() <= 1.5:                # normalised coords -> pixels
            boxes = boxes * self.size
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale    # undo letterbox
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
        h, w = img.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        out = []
        for b, s, c in zip(boxes, scores, classes):
            if (b[2] - b[0]) * (b[3] - b[1]) < self.min_box_frac * h * w:
                continue
            name = self.class_names[c] if self.class_names and c < len(self.class_names) else str(int(c))
            out.append({"box": tuple(float(v) for v in b), "conf": float(s), "label": name})
        out.sort(key=lambda d: -d["conf"])
        return {"boxes": out, "best_score": float(best)}

    # ------------------------------------------------------------------ annotated image
    COLOR_WOUND = (230, 40, 40)
    COLOR_OK = (30, 160, 70)

    @staticmethod
    def _font(size, bold=False):
        names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        dirs = ["/usr/share/fonts/truetype/dejavu/", ""]
        for d in dirs:
            for n in names:
                try:
                    return ImageFont.truetype(d + n, size)
                except (OSError, IOError):
                    pass
        try:
            return ImageFont.load_default(size=size)          # Pillow >= 10.1
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _text_size(draw, text, font):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return x1 - x0, y1 - y0

    def _draw_panel(self, img, boxes, title, height):
        """One foot image -> PIL image of fixed height with boxes, box labels and a title bar."""
        pil = Image.fromarray(img)
        scale = height / pil.height
        pil = pil.resize((max(1, int(round(pil.width * scale))), height), Image.BILINEAR)
        d = ImageDraw.Draw(pil)
        lw = max(2, height // 150)
        f_box = self._font(max(12, height // 28), bold=True)
        for b in boxes:
            x1, y1, x2, y2 = [v * scale for v in b["box"]]
            d.rectangle([x1, y1, x2, y2], outline=self.COLOR_WOUND, width=lw)
            txt = f"{b['label']} {b['conf']:.2f}"
            tw, th = self._text_size(d, txt, f_box)
            ty = y1 - th - 2 * lw if y1 - th - 2 * lw >= 0 else y1 + lw   # above box, or inside if no room
            d.rectangle([x1, ty, x1 + tw + 2 * lw, ty + th + 2 * lw], fill=self.COLOR_WOUND)
            d.text((x1 + lw, ty + lw - d.textbbox((0, 0), txt, font=f_box)[1]), txt,
                   fill=(255, 255, 255), font=f_box)
        # title bar on top of the panel
        f_t = self._font(max(12, height // 22), bold=True)
        n = len(boxes)
        title = f"{title}: {n} wound{'s' if n != 1 else ''}" if n else f"{title}: no wound"
        tw, th = self._text_size(d, title, f_t)
        pad = max(4, height // 80)
        bar = Image.new("RGB", (pil.width, th + 2 * pad), self.COLOR_WOUND if n else self.COLOR_OK)
        ImageDraw.Draw(bar).text((pad, pad - d.textbbox((0, 0), title, font=f_t)[1]), title,
                                 fill=(255, 255, 255), font=f_t)
        out = Image.new("RGB", (pil.width, bar.height + pil.height))
        out.paste(bar, (0, 0)); out.paste(pil, (0, bar.height))
        return out

    def _save_annotated(self, panels, confidence, label, save_path, file_prefix=None, panel_height=480):
        if Image is None:
            raise RuntimeError("Saving images needs Pillow: sudo apt install python3-pil")
        stem = f"{file_prefix}_wound" if file_prefix else "wound"
        save_path = _output_path(save_path, stem, ".jpg", (".jpg", ".jpeg", ".png"))
        h = min(panel_height, max(p[0].shape[0] for p in panels))
        tiles = [self._draw_panel(img, boxes, title, h) for img, boxes, title in panels]
        gap = max(4, h // 60)
        W = sum(t.width for t in tiles) + gap * (len(tiles) + 1)
        f_big, f_small = self._font(max(16, h // 14), bold=True), self._font(max(11, h // 30))
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        header = f"{label}   ({confidence:.1%} confidence)"
        footer = (f"{datetime.now():%Y-%m-%d %H:%M:%S}   |   threshold {self.conf_threshold:.2f}   |   "
                  f"boxes: " + ", ".join(f"{t.split()[0].lower()} {len(b)}" for _, b, t in panels))
        _, hh = self._text_size(tmp, header, f_big)
        _, fh = self._text_size(tmp, footer, f_small)
        pad = max(6, h // 40)
        H_head, H_foot = hh + 2 * pad, fh + 2 * pad
        W = max(W, self._text_size(tmp, header, f_big)[0] + 2 * pad,
                self._text_size(tmp, footer, f_small)[0] + 2 * pad)
        th = max(t.height for t in tiles)
        canvas = Image.new("RGB", (W, H_head + th + gap * 2 + H_foot), (25, 25, 25))
        d = ImageDraw.Draw(canvas)
        positive = label == self.LABEL_POSITIVE
        d.rectangle([0, 0, W, H_head], fill=self.COLOR_WOUND if positive else self.COLOR_OK)
        d.text((pad, pad - d.textbbox((0, 0), header, font=f_big)[1]), header, fill=(255, 255, 255), font=f_big)
        x = gap
        for t in tiles:
            canvas.paste(t, (x, H_head + gap)); x += t.width + gap
        fy = H_head + th + 2 * gap
        d.text((pad, fy + pad - d.textbbox((0, 0), footer, font=f_small)[1]), footer,
               fill=(220, 220, 220), font=f_small)
        if save_path.lower().endswith((".jpg", ".jpeg")):
            canvas.save(save_path, quality=90)
        else:
            canvas.save(save_path)
        return os.path.abspath(save_path)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        if Image is not None:
            resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        else:                                                 # numpy nearest-neighbour fallback
            ys = (np.arange(nh) / scale).astype(int).clip(0, h - 1)
            xs = (np.arange(nw) / scale).astype(int).clip(0, w - 1)
            resized = img[ys][:, xs]
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        py, px = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, (px, py)

    def _decode(self, y):
        """Supports both Ultralytics output styles. Returns boxes, scores, classes, best raw score."""
        if y.ndim == 2 and y.shape[-1] == 6:                  # end-to-end / NMS: x1,y1,x2,y2,score,cls
            best = float(y[:, 4].max()) if len(y) else 0.0
            k = y[:, 4] >= self.conf_threshold
            return y[k, :4].copy(), y[k, 4], y[k, 5].astype(int), best
        if y.shape[0] < y.shape[1]:                           # raw (4+nc, N) -> (N, 4+nc)
            y = y.T
        cls_scores = y[:, 4:]
        classes, scores = cls_scores.argmax(1), cls_scores.max(1)
        best = float(scores.max()) if len(scores) else 0.0
        k = scores >= self.conf_threshold
        b, scores, classes = y[k, :4], scores[k], classes[k]
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                          b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)   # cx,cy,w,h -> corners
        keep = self._nms(boxes, scores)
        return boxes[keep], scores[keep], classes[keep], best

    def _nms(self, b, s):
        order, keep = s.argsort()[::-1], []
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        while order.size:
            i = order[0]; keep.append(i)
            rest = order[1:]
            xx1 = np.maximum(b[i, 0], b[rest, 0]); yy1 = np.maximum(b[i, 1], b[rest, 1])
            xx2 = np.minimum(b[i, 2], b[rest, 2]); yy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            order = rest[inter / (area[i] + area[rest] - inter + 1e-9) < self.iou_threshold]
        return np.array(keep, dtype=int)


if __name__ == "__main__":
    # demo:  python3 wound_predictor.py wound.tflite left.jpg right.jpg
    import sys
    model = WoundPredictor(sys.argv[1], class_names=["wound"])
    left, right = (np.asarray(Image.open(p).convert("RGB")) for p in sys.argv[2:4])
    out = sys.argv[4] if len(sys.argv) > 4 else None          # optional: folder for annotated image
    conf, label, info = model.predict(left, right, save_path=out, return_details=True)
    print(f"{label}  (confidence {conf:.1%})  left={info['left_wound']} right={info['right_wound']}")
    if info["saved_to"]:
        print("saved:", info["saved_to"])