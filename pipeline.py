from __future__ import annotations

import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from capture import ScreenCapture
from config import Settings
from ocr import OCR, Block, get_ocr
from tracker import GRID, Tracker, downscale, make_template
from translator import make_translator

MAX_WAIT = 1.0
REFRESH = 20.0
REFRESH_BATCH = 8


def _overlap(a: Block, b: Block) -> bool:
    w = min(a.right, b.right) - max(a.x, b.x)
    h = min(a.bottom, b.bottom) - max(a.y, b.y)
    return w > 0 and h > 0 and w * h > 0.3 * min(a.w * a.h, b.w * b.h)


class Pipeline(QThread):
    blocks_ready = Signal(list)
    status = Signal(str)
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, s: Settings):
        super().__init__()
        self.s = s
        self.region = s.region
        self.active = True
        self._running = True
        self._reset = True

    def set_region(self, region):
        self.region = region or None
        self._reset = True

    def set_active(self, on: bool):
        self.active = on
        self._reset = True
        if not on:
            self.blocks_ready.emit([])

    def stop(self):
        self._running = False
        self.wait(3000)

    def _emit(self, blocks: list[Block]):
        self.blocks_ready.emit([replace(b, tpl=None) for b in blocks])

    def _recognize(self, ocr: OCR, tr, frame: np.ndarray, small: np.ndarray, ox: int, oy: int,
                   known: list[tuple[int, int, int, int]]) -> list[Block]:
        t0 = time.perf_counter()
        found = ocr.read(frame, merge=self.s.merge_lines, known=known)
        t1 = time.perf_counter()
        now = time.monotonic()
        for b in found:
            b.x += ox
            b.y += oy
            b.born = now
        for b in (b for b in found if b.junk):
            b.tpl = make_template(small, b, ox, oy)
        real = [b for b in found if not b.junk]
        for b, t in zip(real, tr.translate_many([b.text for b in real])):
            b.translated = t if t != b.text else ""
            b.tpl = make_template(small, b, ox, oy)
        t2 = time.perf_counter()
        print(f"[pipeline] новых блоков {len(real)} + мусор {len(found) - len(real)} (известных {len(known)}) | "
              f"OCR {1000*(t1-t0):.0f} мс | перевод {1000*(t2-t1):.0f} мс")
        return found

    def run(self):
        self.status.emit("Загрузка модели распознавания…")
        try:
            cap = ScreenCapture(self.s.monitor)
            ocr = get_ocr(self.s.ocr, self.s.src, self.s.gpu, self.s.min_score, self.s.min_height,
                          lambda i, n: self.status.emit(f"Подготовка видеокарты: {i}/{n}"))
            tr = make_translator(self.s.translator, self.s.src, self.s.dst)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
            return
        if not self._running:
            cap.close()
            return
        self.status.emit("")
        self.ready.emit(ocr.device)
        print(f"[pipeline] OCR на {ocr.device}")

        pool = ThreadPoolExecutor(max_workers=1)
        tracker = Tracker()
        job: Future | None = None
        job_region, job_epoch, job_small = None, 0, None
        epoch = 0
        prev = None
        blocks: list[Block] = []
        dirty, last_change, dirty_since = True, 0.0, 0.0
        while self._running:
            time.sleep(self.s.interval)
            if not self.active:
                continue
            region = self.region
            if self._reset:
                self._reset, prev, blocks, dirty, job = False, None, [], True, None
                self.blocks_ready.emit([])
            frame, changed = cap.grab(region)
            if frame is None:
                continue
            ox, oy = (region[0], region[1]) if region else (0, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = downscale(gray)
            grid = cv2.resize(gray, (max(gray.shape[1] // GRID, 1), max(gray.shape[0] // GRID, 1)),
                              interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            moved = False

            if changed and prev is not None and prev.shape == grid.shape:
                mask = cv2.absdiff(grid, prev) > 18
                if mask.mean() > 0.0005:
                    if not dirty:
                        dirty_since = now
                    dirty, last_change = True, now
                    epoch += 1
                    blocks, moved = tracker.update(blocks, small, mask, ox, oy)
            prev = grid

            if job is not None and job.done():
                try:
                    found = job.result()
                except Exception as e:
                    print(f"[pipeline] ошибка OCR: {e}")
                    found = None
                if found is not None and job_region == region:
                    if job_epoch != epoch:
                        found, _ = tracker.update(found, small, None, ox, oy, ref=job_small)
                    blocks = [b for b in blocks if not any(_overlap(b, n) for n in found)] + found
                    moved = moved or bool(found)
                job = None

            if moved:
                self._emit(blocks)

            if job is None and dirty and (now - last_change >= self.s.settle or now - dirty_since >= MAX_WAIT):
                dirty, dirty_since = False, now
                job_region, job_epoch, job_small = region, epoch, small
                stale = {id(b) for b in sorted((b for b in blocks if now - b.born >= REFRESH),
                                               key=lambda b: b.born)[:REFRESH_BATCH]}
                known = [(b.x - ox, b.y - oy, b.right - ox, b.bottom - oy)
                         for b in blocks if id(b) not in stale]
                job = pool.submit(self._recognize, ocr, tr, np.ascontiguousarray(frame.copy()),
                                  small, ox, oy, known)
        pool.shutdown(wait=False, cancel_futures=True)
        cap.close()
