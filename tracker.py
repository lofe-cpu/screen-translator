from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from ocr import Block

SCALE = 2
SEARCH_X = 80
SEARCH_Y = 220
MATCH_MIN = 0.80
SOLO_MIN = 0.97
DIST_PENALTY = 0.0003
GRID = 8


def downscale(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    return cv2.resize(gray, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)


def make_template(small: np.ndarray, b: Block, ox: int, oy: int) -> np.ndarray | None:
    x0, y0 = (b.x - ox) // SCALE, (b.y - oy) // SCALE
    tpl = small[max(y0, 0):y0 + max(b.h // SCALE, 1), max(x0, 0):x0 + max(b.w // SCALE, 1)]
    if tpl.shape[0] < 4 or tpl.shape[1] < 4 or tpl.std() < 4:
        return None
    return tpl.copy()


def _score_at(small: np.ndarray, tpl: np.ndarray, x: int, y: int) -> float:
    th, tw = tpl.shape
    if x < 0 or y < 0 or y + th > small.shape[0] or x + tw > small.shape[1]:
        return -1.0
    r = cv2.matchTemplate(small[y:y + th, x:x + tw], tpl, cv2.TM_CCOEFF_NORMED)[0, 0]
    return float(r) if np.isfinite(r) else -1.0


def _search(small: np.ndarray, tpl: np.ndarray, x0: int, y0: int) -> tuple[int, int, float] | None:
    th, tw = tpl.shape
    H, W = small.shape
    sx0, sy0 = max(x0 - SEARCH_X, 0), max(y0 - SEARCH_Y, 0)
    sx1, sy1 = min(x0 + tw + SEARCH_X, W), min(y0 + th + SEARCH_Y, H)
    win = small[sy0:sy1, sx0:sx1]
    if win.shape[0] < th or win.shape[1] < tw:
        return None
    res = np.nan_to_num(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED), nan=-1.0)
    ys, xs = np.ogrid[:res.shape[0], :res.shape[1]]
    score = res - DIST_PENALTY * np.hypot(xs + sx0 - x0, ys + sy0 - y0)
    _, _, _, (bx, by) = cv2.minMaxLoc(score)
    if res[by, bx] < MATCH_MIN:
        return None
    return bx + sx0, by + sy0, float(res[by, bx])


class Tracker:
    def __init__(self):
        self._ref: np.ndarray | None = None
        self._last_delta = (0, 0)

    def _candidate(self, small, tpl, x0, y0, guesses) -> tuple[int, int, float] | None:
        for dx, dy in guesses:
            s = _score_at(small, tpl, x0 + dx, y0 + dy)
            if s >= MATCH_MIN:
                return dx, dy, s
        found = _search(small, tpl, x0, y0)
        return None if found is None else (found[0] - x0, found[1] - y0, found[2])

    def update(self, blocks: list[Block], small: np.ndarray, mask: np.ndarray | None,
               ox: int, oy: int, ref: np.ndarray | None = None) -> tuple[list[Block], bool]:
        ref = self._ref if ref is None else ref
        self._ref = small
        keep, todo = [], []
        for b in blocks:
            if mask is not None and not _touches(mask, b, ox, oy):
                keep.append(b)
            elif b.tpl is not None:
                todo.append((b, (b.x - ox) // SCALE, (b.y - oy) // SCALE))
        changed = len(keep) + len(todo) != len(blocks)
        if not todo:
            return keep, changed

        shift = _global_shift(ref, small)
        guesses = list(dict.fromkeys(g for g in ((0, 0), shift, self._last_delta) if g is not None))
        cands = [(b, x0, y0, self._candidate(small, b.tpl, x0, y0, guesses)) for b, x0, y0 in todo]

        votes = Counter((c[0], c[1]) for *_, c in cands if c is not None)
        trusted = {d for d in (shift, votes.most_common(1)[0][0] if votes else None) if d is not None}
        moving = [d for d in trusted if d != (0, 0)]
        if moving:
            self._last_delta = moving[0]

        for b, x0, y0, c in cands:
            if c is not None and (c[0], c[1]) not in trusted and trusted:
                scores = {d: _score_at(small, b.tpl, x0 + d[0], y0 + d[1]) for d in trusted}
                best = max(scores, key=scores.get)
                if scores[best] >= MATCH_MIN:
                    c = (best[0], best[1], scores[best])
                elif -1.0 in scores.values() or c[2] < SOLO_MIN:
                    c = None
            if c is None:
                changed = True
                continue
            if c[0] or c[1]:
                b.x += c[0] * SCALE
                b.y += c[1] * SCALE
                changed = True
            keep.append(b)
        return keep, changed


def _global_shift(a: np.ndarray | None, b: np.ndarray) -> tuple[int, int] | None:
    if a is None or a.shape != b.shape:
        return None
    (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    if response < 0.05:
        return None
    return int(round(dx)), int(round(dy))


def _touches(mask: np.ndarray, b: Block, ox: int, oy: int) -> bool:
    x0, y0 = (b.x - ox) // GRID, (b.y - oy) // GRID
    x1, y1 = (b.right - ox) // GRID + 1, (b.bottom - oy) // GRID + 1
    return bool(mask[max(y0, 0):y1, max(x0, 0):x1].any())
