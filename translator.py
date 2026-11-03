from __future__ import annotations

import threading

import requests


class Translator:
    def __init__(self, src: str, dst: str):
        self.src, self.dst = src, dst
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def translate_many(self, texts: list[str]) -> list[str]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            try:
                out = self._translate_batch(todo)
            except Exception as e:
                print(f"[translate] ошибка: {e}")
                out = todo
            with self._lock:
                self._cache.update(zip(todo, out))
                if len(self._cache) > 5000:
                    self._cache.clear()
        return [self._cache.get(t, t) for t in texts]

    def _translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError


class GoogleTranslator(Translator):
    BATCH_URL = "https://clients5.google.com/translate_a/t"
    GTX_URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, src: str, dst: str):
        super().__init__(src, dst)
        self._session = requests.Session()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        out: list[str] = []
        chunk: list[str] = []
        size = 0
        for t in texts + [None]:
            if t is None or size + len(t) > 4000:
                if chunk:
                    try:
                        out.extend(self._batch(chunk))
                    except Exception as e:
                        print(f"[translate] пакетный запрос не прошёл ({e}), пробую gtx")
                        out.extend(self._gtx(t) for t in chunk)
                chunk, size = [], 0
            if t is not None:
                chunk.append(t)
                size += len(t) + 1
        return out

    def _batch(self, chunk: list[str]) -> list[str]:
        r = self._session.post(self.BATCH_URL,
                               params={"client": "dict-chrome-ex", "sl": self.src, "tl": self.dst},
                               data={"q": chunk}, timeout=10)
        r.raise_for_status()
        res = r.json()
        if len(chunk) == 1 and isinstance(res, list) and res and isinstance(res[0], str):
            res = [res]
        out = [x[0] if isinstance(x, list) else x for x in res]
        if len(out) != len(chunk):
            raise ValueError(f"ожидалось {len(chunk)} строк, пришло {len(out)}")
        return out

    def _gtx(self, text: str) -> str:
        r = self._session.post(self.GTX_URL,
                               params={"client": "gtx", "sl": self.src, "tl": self.dst, "dt": "t"},
                               data={"q": text}, timeout=10)
        r.raise_for_status()
        return "".join(seg[0] for seg in r.json()[0] if seg and seg[0])


class ArgosTranslator(Translator):
    def __init__(self, src: str, dst: str):
        if src == "auto":
            raise ValueError("Argos не умеет определять язык: укажите --src явно")
        super().__init__(src, dst)
        import argostranslate.translate as at
        langs = {l.code: l for l in at.get_installed_languages()}
        if src not in langs or dst not in langs:
            raise RuntimeError(f"Нет модели Argos {src}->{dst}. Запустите с --install-argos")
        self._tr = langs[src].get_translation(langs[dst])
        if self._tr is None:
            raise RuntimeError(f"Нет модели Argos {src}->{dst}. Запустите с --install-argos")

    def _translate_batch(self, texts: list[str]) -> list[str]:
        return [self._tr.translate(t) for t in texts]


def install_argos(src: str, dst: str):
    import argostranslate.package as ap
    ap.update_package_index()
    pkgs = ap.get_available_packages()
    pairs = [(src, dst)]
    if not any(p.from_code == src and p.to_code == dst for p in pkgs):
        pairs = [(src, "en"), ("en", dst)]
    for f, t in pairs:
        pkg = next((p for p in pkgs if p.from_code == f and p.to_code == t), None)
        if pkg is None:
            print(f"Модель {f}->{t} не найдена")
            continue
        print(f"Скачиваю {f}->{t} ...")
        ap.install_from_path(pkg.download())
    print("Готово")


def make_translator(name: str, src: str, dst: str) -> Translator:
    if name == "argos":
        return ArgosTranslator(src, dst)
    return GoogleTranslator(src, dst)
