from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CYRILLIC = {"ru", "uk", "be", "bg", "kk", "ky", "mn", "sr", "tg", "tt", "ba", "cv", "mk"}
LATIN = {"fr", "de", "es", "it", "pt", "nl", "pl", "cs", "sk", "sl", "hr", "ro", "hu", "tr",
         "sv", "no", "da", "fi", "et", "lv", "lt", "id", "ms", "vi", "uz", "az", "af", "la",
         "ga", "cy", "is", "sq", "tl", "sw"}
ARABIC = {"ar", "fa", "ur", "ug", "ps", "ku"}
DEVANAGARI = {"hi", "mr", "ne", "sa", "bh", "mai"}
CJK = ("zh", "ja", "ko", "auto")


@dataclass
class Block:
    x: int
    y: int
    w: int
    h: int
    text: str
    score: float = 1.0
    translated: str = ""
    bg: tuple[int, int, int] = (255, 255, 255)
    fg: tuple[int, int, int] = (0, 0, 0)
    line_h: float = 0.0
    born: float = 0.0
    junk: bool = False
    tpl: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h


def _script(lang: str) -> str:
    if lang in ("auto", "zh", "zh-cn", "zh-tw"):
        return "ch"
    if lang == "en":
        return "en"
    if lang == "ja":
        return "japan"
    if lang == "ko":
        return "korean"
    if lang in CYRILLIC:
        return "cyrillic"
    if lang in LATIN:
        return "latin"
    if lang in ARABIC:
        return "arabic"
    if lang in DEVANAGARI:
        return "devanagari"
    if lang in ("th", "el", "ta", "te", "ka"):
        return lang
    return "ch"


REC_BATCH = 16
REC_W_STEP = 128
REC_W_MAX = 2048
CUDNN_SEARCH = "EXHAUSTIVE"


class BucketedSession:

    def __init__(self, session, batch: int):
        self.session, self.batch = session, batch

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        W = -(-w // REC_W_STEP) * REC_W_STEP
        if (n, w) != (self.batch, W):
            padded = np.zeros((self.batch, c, h, W), np.float32)
            padded[:n, :, :, :w] = x
            x = padded
        return self.session(x)[:n]

    def warmup(self, progress=None):
        widths = range(REC_W_STEP * 3, REC_W_MAX + 1, REC_W_STEP)
        for i, W in enumerate(widths):
            if progress:
                progress(i + 1, len(widths))
            self.session(np.zeros((self.batch, 3, 48, W), np.float32))

    def __getattr__(self, name):
        return getattr(self.session, name)


class RapidBackend:

    def __init__(self, lang: str, gpu: bool):
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        script = _script(lang)
        params = {"Global.use_cls": False, "Det.lang_type": LangDet.CH}
        if script in ("ch", "en", "japan", "latin"):
            params.update({"Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL,
                           "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL,
                           "Rec.lang_type": LangRec.CH})
        else:
            rec_ver = OCRVersion.PPOCRV4 if script == "ka" else OCRVersion.PPOCRV5
            params.update({"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                           "Rec.ocr_version": rec_ver, "Rec.model_type": ModelType.MOBILE,
                           "Rec.lang_type": LangRec(script)})
        import onnxruntime as ort
        providers = ort.get_available_providers()
        self.device = "CPU"
        if gpu and "CUDAExecutionProvider" in providers:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            params.update({"EngineConfig.onnxruntime.use_cuda": True, "Rec.rec_batch_num": REC_BATCH,
                           "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": CUDNN_SEARCH})
            self.device = "GPU (CUDA)"
        elif gpu and "DmlExecutionProvider" in providers:
            params.update({"EngineConfig.onnxruntime.use_dml": True, "Rec.rec_batch_num": REC_BATCH})
            self.device = "GPU (DirectML)"
        self.engine = RapidOCR(params=params)
        self.rec = self.engine._load_rec_model()
        if self.device != "CPU":
            self.rec.session = BucketedSession(self.rec.session, REC_BATCH)

    def warmup(self, progress=None):
        self.engine(np.full((64, 64, 3), 255, np.uint8), use_cls=False, use_rec=False)
        if isinstance(self.rec.session, BucketedSession):
            self.rec.session.warmup(progress)

    def __call__(self, img: np.ndarray, skip=None) -> list[tuple[np.ndarray, str, float]]:
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        det = self.engine(img, use_cls=False, use_rec=False)
        boxes = getattr(det, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        H, W = img.shape[:2]
        items = []
        for box in boxes:
            x0, y0 = np.floor(box.min(0)).astype(int)
            x1, y1 = np.ceil(box.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 < 3 or y1 - y0 < 3 or (skip and skip(x0, y0, x1, y1)):
                continue
            items.append((box, img[y0:y1, x0:x1]))
        if not items:
            return []
        res = self.rec(TextRecInput(img=[crop for _, crop in items], return_word_box=False))
        return [(box, t, s) for (box, _), t, s in zip(items, res.txts, res.scores)]


class PaddleBackend:
    def __init__(self, lang: str, gpu: bool):
        from paddleocr import PaddleOCR
        plang = {"auto": "ch", "zh": "ch", "ja": "japan", "ko": "korean"}.get(lang, lang)
        self.engine = PaddleOCR(lang=plang, device="gpu" if gpu else "cpu",
                                use_doc_orientation_classify=False, use_doc_unwarping=False,
                                use_textline_orientation=False)

    def __call__(self, img: np.ndarray, skip=None) -> list[tuple[np.ndarray, str, float]]:
        out = []
        for r in self.engine.predict(img):
            out += list(zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]))
        return out


class OCR:
    def __init__(self, backend: str, lang: str, gpu: bool, min_score: float, min_height: int):
        self.lang = lang
        self.min_score, self.min_height = min_score, min_height
        self.engine = PaddleBackend(lang, gpu) if backend == "paddle" else RapidBackend(lang, gpu)
        self.device = getattr(self.engine, "device", "GPU" if gpu else "CPU")

    def warmup(self, progress=None):
        if hasattr(self.engine, "warmup"):
            self.engine.warmup(progress)

    def read(self, img: np.ndarray, merge: bool = True,
             known: list[tuple[int, int, int, int]] = ()) -> list[Block]:
        def skip(x0, y0, x1, y1):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            return any(k[0] <= cx <= k[2] and k[1] <= cy <= k[3] for k in known)

        lines, junk = [], []
        for poly, text, score in self.engine(img, skip if known else None):
            text = text.strip()
            poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            b = Block(int(x0), int(y0), int(x1 - x0), int(y1 - y0), text, float(score), line_h=float(y1 - y0))
            letters = sum(c.isalpha() for c in text)
            if score < self.min_score or y1 - y0 < self.min_height or letters < 2:
                b.junk = True
                junk.append(b)
            else:
                lines.append(b)
        blocks = merge_lines(lines, self.lang) if merge else lines
        for b in blocks:
            b.bg, b.fg = sample_colors(img, b)
        return blocks + junk


_ocr_cache: dict[tuple, OCR] = {}


def get_ocr(backend: str, lang: str, gpu: bool, min_score: float, min_height: int, progress=None) -> OCR:
    key = (backend, _script(lang) if backend == "rapid" else lang, gpu)
    ocr = _ocr_cache.get(key)
    if ocr is None:
        ocr = OCR(backend, lang, gpu, min_score, min_height)
        ocr.warmup(progress)
        _ocr_cache[key] = ocr
    ocr.lang, ocr.min_score, ocr.min_height = lang, min_score, min_height
    return ocr


def _join(a: str, b: str, lang: str) -> str:
    if lang in ("zh", "ja") or (_is_cjk(a[-1:]) and _is_cjk(b[:1])):
        return a + b
    if a.endswith("-") and len(a) > 1 and a[-2].isalpha():
        return a[:-1] + b
    return a + " " + b


def _is_cjk(ch: str) -> bool:
    return bool(ch) and ("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" or "가" <= ch <= "힯")


def merge_lines(lines: list[Block], lang: str) -> list[Block]:
    lines.sort(key=lambda b: (b.y, b.x))
    paras: list[Block] = []
    for ln in lines:
        target = None
        for p in reversed(paras[-8:]):
            h = p.line_h
            gap = ln.y - p.bottom
            same_size = abs(ln.h - h) < 0.35 * max(h, ln.h)
            close = -0.3 * h <= gap <= 0.9 * h
            aligned = abs(ln.x - p.x) < 1.5 * h or (ln.x < p.right and ln.right > p.x)
            if same_size and close and aligned:
                target = p
                break
        if target is None:
            paras.append(ln)
            continue
        x0, y0 = min(target.x, ln.x), min(target.y, ln.y)
        x1, y1 = max(target.right, ln.right), max(target.bottom, ln.bottom)
        target.x, target.y, target.w, target.h = x0, y0, x1 - x0, y1 - y0
        target.text = _join(target.text, ln.text, lang)
        target.score = min(target.score, ln.score)
    return paras


def sample_colors(img: np.ndarray, b: Block) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    H, W = img.shape[:2]
    x0, y0 = max(b.x - 3, 0), max(b.y - 3, 0)
    x1, y1 = min(b.right + 3, W - 1), min(b.bottom + 3, H - 1)
    border = np.concatenate([img[y0, x0:x1], img[y1, x0:x1], img[y0:y1, x0], img[y0:y1, x1]])
    if len(border) == 0:
        return (255, 255, 255), (0, 0, 0)
    bgr = np.median(border, axis=0)
    rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return rgb, ((20, 20, 20) if lum > 140 else (245, 245, 245))
