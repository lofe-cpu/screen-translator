from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = SPECPATH + "\\.."

datas, binaries, hiddenimports = [], [], []
for pkg in ("rapidocr", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("dxcam") + collect_submodules("comtypes") + ["keyboard", "mss"]

a = Analysis(
    [ROOT + "\\main.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "torch", "paddle", "paddleocr",
              "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ScreenTranslator",
    icon=SPECPATH + "\\icon.ico",
    console=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ScreenTranslator")
