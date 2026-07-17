from PIL import Image

from caricature_mvp.image_io import normalize_image


def test_resize_preserves_aspect_ratio_and_does_not_crop():
    source = Image.new("RGB", (1600, 800), "red")
    normalized = normalize_image(source, max_side=1000, alignment=1)
    assert normalized.size == (1000, 500)
    assert normalized.getpixel((0, 0)) == (255, 0, 0)
    assert normalized.getpixel((999, 499)) == (255, 0, 0)


def test_alignment_adds_only_white_padding():
    source = Image.new("RGB", (101, 51), "blue")
    normalized = normalize_image(source, max_side=1024, alignment=16)
    assert normalized.size == (112, 64)
    blue_pixels = sum(pixel == (0, 0, 255) for pixel in normalized.getdata())
    assert blue_pixels == 101 * 51
