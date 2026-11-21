"""Wound detector for RGB foot images on Raspberry Pi (YOLO exported to LiteRT / .tflite).

    from wound_predictor import WoundPredictor
    model = WoundPredictor("wound.tflite", class_names=["wound"])
    confidence, label = model.predict(left_rgb, right_rgb)          # both feet
    confidence, label = model.predict_single(foot_rgb)              # one image
    confidence, label, info = model.predict(left_rgb, right_rgb, return_details=True)
"""
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

try:
    from PIL import Image                                   # bilinear resize, same as YOLO training
except ImportError:
    Image = None


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
    def predict(self, left, right, return_details=False):
        """left, right: HxWx3 uint8 images of each foot.
        Returns (confidence, label). Wound found -> confidence = best wound score;
        no wound -> confidence = 1 - best (rejected) score."""
        res_l, res_r = self._run(left), self._run(right)
        boxes = res_l["boxes"] + res_r["boxes"]
        best = max(res_l["best_score"], res_r["best_score"])
        confidence, label = self._decide(boxes, best)
        if return_details:
            return confidence, label, {"left": res_l["boxes"], "right": res_r["boxes"],
                                       "left_wound": bool(res_l["boxes"]), "right_wound": bool(res_r["boxes"]),
                                       "best_score": best, "conf_threshold": self.conf_threshold}
        return confidence, label

    def predict_single(self, image, return_details=False):
        res = self._run(image)
        confidence, label = self._decide(res["boxes"], res["best_score"])
        if return_details:
            return confidence, label, {"boxes": res["boxes"], "best_score": res["best_score"]}
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

    def _run(self, image):
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        if self.input_is_bgr:
            img = img[..., ::-1]
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)

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


# if __name__ == "__main__":
#     # demo:  python3 wound_predictor.py wound.tflite left.jpg right.jpg
#     import sys
#     model = WoundPredictor(sys.argv[1], class_names=["wound"])
#     left, right = (np.asarray(Image.open(p).convert("RGB")) for p in sys.argv[2:4])
#     conf, label, info = model.predict(left, right, return_details=True)
#     print(f"{label}  (confidence {conf:.1%})  left={info['left_wound']} right={info['right_wound']}")