@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

if not exist build\venv\Scripts\python.exe (
    python -m venv build\venv || exit /b 1
    build\venv\Scripts\python -m pip install --upgrade pip
    build\venv\Scripts\python -m pip install PySide6-Essentials dxcam mss numpy opencv-python rapidocr ^
        onnxruntime-directml requests keyboard pyinstaller pillow || exit /b 1
)

build\venv\Scripts\python build\prefetch_models.py || exit /b 1
build\venv\Scripts\python build\make_icon.py || exit /b 1
build\venv\Scripts\pyinstaller --noconfirm --clean --distpath dist --workpath build\work build\ScreenTranslator.spec || exit /b 1

powershell -NoProfile -Command "Compress-Archive -Force -Path dist\ScreenTranslator -DestinationPath dist\ScreenTranslator.zip"
echo.
echo Готово: dist\ScreenTranslator\ScreenTranslator.exe  и  dist\ScreenTranslator.zip
