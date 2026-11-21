"""Foot thermogram classifier for Raspberry Pi (LiteRT).

    from foot_predictor import FootPredictor
    model = FootPredictor("foot_model.tflite", "pi_config.json")
    confidence, label = model.predict(left_temps, right_temps)   # 2D numpy arrays in °C
    confidence, label = model.predict_frame(mlx_frame)           # one 24x32 frame with both feet
    model.predict(left_temps, right_temps, save_path="results")   # -> results/thermal_20261121_234700.txt
"""
import os
import json
from datetime import datetime
import numpy as np
from scipy import ndimage

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


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


class FootPredictor:
    LABEL_POSITIVE = "DM / High Risk (Ulcer)"
    LABEL_NEGATIVE = "Control / Healthy"

    def __init__(self, model_path, config_path, threshold=None, num_threads=4,
                 swap_frame_sides=False, min_foot_pixels=50):
        """
        model_path       : path to foot_model.tflite
        config_path      : path to pi_config.json (features, medians, threshold, foot_size)
        threshold        : override the decision threshold (None = value from config).
                           Lower it (e.g. 0.5) to catch more high-risk feet at the cost of more false alarms.
        num_threads      : CPU threads for LiteRT (Pi 3 has 4 cores)
        swap_frame_sides : for predict_frame(): True if the patient's LEFT foot is on the RIGHT of the frame
        min_foot_pixels  : minimum warm pixels needed to accept a foot (lower it for tiny MLX crops)
        """
        with open(config_path) as f:
            cfg = json.load(f)
        self.features = cfg["features"]
        self.medians = np.array(cfg["medians"], dtype=np.float32)
        self.foot_size = tuple(cfg["foot_size"])
        self.threshold = float(cfg["threshold"] if threshold is None else threshold)
        self.swap_frame_sides = swap_frame_sides
        self.min_foot_pixels = min_foot_pixels

        self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interp.allocate_tensors()
        ins = {len(d["shape"]): d["index"] for d in self.interp.get_input_details()}
        self._img_idx, self._feat_idx = ins[4], ins[2]          # 4D image, 2D feature vector
        self._out_idx = self.interp.get_output_details()[0]["index"]

    # ------------------------------------------------------------------ public API
    def predict(self, left, right, save_path=None, return_details=False, file_prefix=None):
        """left, right: 2D numpy arrays of temperatures (°C, or Kelvin - auto-converted).
        save_path: folder to save a .txt report in (file name generated with a timestamp,
                   e.g. thermal_20261121_234700.txt). None = don't save.
        file_prefix: optional start of the file name, e.g. "patient01" -> patient01_thermal_<time>.txt
        Returns (confidence, label); confidence = probability of the returned label (0.5-1.0)."""
        p_dm, feats = self._p_dm(left, right)
        is_pos = p_dm >= self.threshold
        label = self.LABEL_POSITIVE if is_pos else self.LABEL_NEGATIVE
        confidence = p_dm if is_pos else 1.0 - p_dm
        details = {"p_dm": p_dm, "threshold": self.threshold,
                   "max_lr_diff_C": float(feats["reg_max_absdiff"]),
                   "asymmetry_flag": bool(feats["reg_max_absdiff"] > 2.2),
                   "left_mean_C": float(feats["L_mean"]), "right_mean_C": float(feats["R_mean"]),
                   "saved_to": None}
        if save_path:
            details["saved_to"] = self._save_report(save_path, confidence, label, details, file_prefix)
        if return_details:
            return confidence, label, details
        return confidence, label

    def predict_frame(self, frame, save_path=None, return_details=False, file_prefix=None):
        """frame: one 2D matrix containing both feet side by side (e.g. MLX90640 24x32)."""
        frame = np.asarray(frame, dtype=float)
        h = frame.shape[1] // 2
        left, right = frame[:, :h], frame[:, h:]
        if self.swap_frame_sides:
            left, right = right, left
        return self.predict(left, right, save_path, return_details, file_prefix)

    # ------------------------------------------------------------------ text report
    def _save_report(self, save_path, confidence, label, d, file_prefix=None):
        stem = f"{file_prefix}_thermal" if file_prefix else "thermal"
        save_path = _output_path(save_path, stem, ".txt", (".txt", ".log"))
        lines = [
            "THERMAL FOOT ANALYSIS",
            "=" * 40,
            f"Date/time          : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Prediction         : {'POSITIVE' if label == self.LABEL_POSITIVE else 'NEGATIVE'}",
            f"Label              : {label}",
            f"Confidence         : {confidence:.1%}",
            f"P(DM pattern)      : {d['p_dm']:.4f}",
            f"Decision threshold : {d['threshold']:.2f}",
            "-" * 40,
            f"Left foot mean     : {d['left_mean_C']:.2f} °C",
            f"Right foot mean    : {d['right_mean_C']:.2f} °C",
            f"Max L-R difference : {d['max_lr_diff_C']:.2f} °C (regional)",
            f"Asymmetry > 2.2 °C : {'YES - check this foot clinically' if d['asymmetry_flag'] else 'no'}",
            "",
        ]
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return os.path.abspath(save_path)

    # ------------------------------------------------------------------ inference
    def _p_dm(self, left, right):
        img, f = self._make_sample(left, right)
        x = np.array([f[k] for k in self.features], dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.medians)            # same imputation as training
        self.interp.set_tensor(self._img_idx, img[None])
        self.interp.set_tensor(self._feat_idx, x[None].astype(np.float32))
        self.interp.invoke()
        return float(self.interp.get_tensor(self._out_idx).ravel()[0]), f

    # ------------------------------------------------------------------ preprocessing (same as training)
    @staticmethod
    def _otsu(v, bins=256):
        hist, edges = np.histogram(v, bins=bins)
        c = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(hist); w1 = w0[-1] - w0
        s0 = np.cumsum(hist * c)
        m0 = s0 / np.maximum(w0, 1); m1 = (s0[-1] - s0) / np.maximum(w1, 1)
        k = np.argmax(w0 * w1 * (m0 - m1) ** 2)
        return c[k], m0[k], m1[k]

    def _segment(self, T):
        T = np.array(T, dtype=float)
        if np.nanmedian(T) > 200:                                # Kelvin -> °C
            T = T - 273.15
        valid = np.isfinite(T) & (T > 10) & (T < 45)
        if valid.sum() < self.min_foot_pixels:
            raise ValueError("no foot found in the temperature matrix")
        if (~valid).mean() > 0.10:                               # background already removed
            mask = valid.copy()
        else:
            thr, lo, hi = self._otsu(T[valid])
            mask = valid & (T > thr) if (hi - lo) > 3.0 else valid.copy()
        mask = ndimage.binary_opening(mask, iterations=1)
        lab, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            mask = lab == (np.argmax(sizes) + 1)
        return T, ndimage.binary_fill_holes(mask)

    @staticmethod
    def _resize(a, size, order=1):
        z = ndimage.zoom(a, (size[0] / a.shape[0], size[1] / a.shape[1]), order=order)
        z = z[:size[0], :size[1]]
        return np.pad(z, ((0, size[0] - z.shape[0]), (0, size[1] - z.shape[1])), mode="edge")

    def _to_grid(self, T, mask):
        ys, xs = np.where(mask)
        T = T[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        Tf = np.where(m, T, T[m].mean())
        return self._resize(Tf, self.foot_size), self._resize(m.astype(float), self.foot_size) > 0.5

    @staticmethod
    def _region_means(T, m, rows=4, cols=2):
        H, W = T.shape; out = []
        for i in range(rows):
            for j in range(cols):
                sl = (slice(i * H // rows, (i + 1) * H // rows), slice(j * W // cols, (j + 1) * W // cols))
                out.append(T[sl][m[sl]].mean() if m[sl].sum() > 3 else np.nan)
        return np.array(out)

    def _features(self, L, ml, R, mr):
        f = {}
        for nm, T, m in (("L", L, ml), ("R", R, mr)):
            v = T[m]
            f[f"{nm}_mean"] = v.mean();  f[f"{nm}_std"] = v.std()
            f[f"{nm}_p5"], f[f"{nm}_p50"], f[f"{nm}_p95"] = np.percentile(v, [5, 50, 95])
            f[f"{nm}_range"] = f[f"{nm}_p95"] - f[f"{nm}_p5"]
        f["mean_both"] = (f["L_mean"] + f["R_mean"]) / 2
        m = ml & mr
        d = (L - R)[m] if m.sum() > 10 else np.array([np.nan])
        f["asym_mean"] = np.mean(d)
        f["asym_absmean"] = np.mean(np.abs(d))
        f["asym_p95"] = np.percentile(np.abs(d), 95)
        f["asym_frac_gt2.2"] = np.mean(np.abs(d) > 2.2)
        rl, rr = self._region_means(L, ml), self._region_means(R, mr)
        for k in range(len(rl)):
            f[f"regL{k}_rel"] = rl[k] - f["L_mean"]
            f[f"regR{k}_rel"] = rr[k] - f["R_mean"]
            f[f"regDiff{k}"] = rl[k] - rr[k]
        f["reg_max_absdiff"] = np.nanmax(np.abs(rl - rr))
        return f

    def _make_sample(self, left, right):
        TL, mL = self._segment(left)
        TR, mR = self._segment(right)
        L, ml = self._to_grid(TL, mL)
        R, mr = self._to_grid(TR, mR)
        R, mr = np.fliplr(R), np.fliplr(mr)                      # mirror right foot onto left
        m = ml & mr
        mu = (L[ml].mean() + R[mr].mean()) / 2                   # remove person's baseline
        img = np.stack([np.where(ml, L - mu, 0), np.where(mr, R - mu, 0),
                        np.where(m, L - R, 0), m.astype(float)], axis=-1).astype(np.float32)
        return img, self._features(L, ml, R, mr)


# if __name__ == "__main__":
#     # quick demo:  python3 foot_predictor.py left.csv right.csv [results_folder]
#     import sys
#     model = FootPredictor("foot_model.tflite", "pi_config.json")
#     left = np.genfromtxt(sys.argv[1], delimiter=",")
#     right = np.genfromtxt(sys.argv[2], delimiter=",")
#     out = sys.argv[3] if len(sys.argv) > 3 else None          # optional: folder for the .txt report
#     conf, label = model.predict(left, right, save_path=out)
#     print(f"{label}  (confidence {conf:.1%})")