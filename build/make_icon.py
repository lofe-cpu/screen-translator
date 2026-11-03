from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).with_name("icon.ico")
FONTS = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\YuGothB.ttc"]


def draw(size: int) -> Image.Image:
    k = 4
    s = size * k
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = s // 16
    d.rounded_rectangle((m, m, s - m, s - m), radius=s // 5, fill=(40, 140, 255, 255))
    font = next((ImageFont.truetype(f, int(s * 0.42)) for f in FONTS if Path(f).exists()),
                ImageFont.load_default())
    d.text((s / 2, s / 2), "文A", font=font, fill="white", anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


sizes = [16, 24, 32, 48, 64, 128, 256]
draw(256).save(OUT, sizes=[(n, n) for n in sizes])
print(OUT)
