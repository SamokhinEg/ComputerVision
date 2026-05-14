import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def generate_test_image(width=400, height=400):
    """Генерує яскраве тестове зображення для обробки, якщо немає власного."""
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    r = (np.sin(xx * 10) * 0.5 + 0.5) * 255
    g = (np.cos(yy * 10) * 0.5 + 0.5) * 255
    b = (np.sin((xx + yy) * 5) * 0.5 + 0.5) * 255

    img = np.stack((r, g, b), axis=-1).astype(np.uint8)
    return img


def apply_gradient_blend(original, filtered, mask):
    """Матричне змішування двох зображень за маскою [0..1]"""
    # Розширюємо маску до 3 каналів (H, W, 1) для матричного множення
    mask_3d = np.expand_dims(mask, axis=-1)
    result = original * (1 - mask_3d) + filtered * mask_3d
    return np.clip(result, 0, 255).astype(np.uint8)


# === 1. ФУНКЦІЇ ФІЛЬТРІВ ===
def filter_grayscale(img):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return np.stack((gray, gray, gray), axis=-1)


def filter_negative(img):
    return 255 - img


def filter_sepia(img):
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    # Матричне множення тензорів: img @ sepia_matrix.T
    sepia = np.dot(img[..., :3], sepia_matrix.T)
    return np.clip(sepia, 0, 255)


# === 2. ФУНКЦІЇ ГРАДІЄНТІВ ===
def gradient_diagonal(h, w):
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return (x + y) / (w + h - 2)


def gradient_from_center(h, w):
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    return dist / max_dist


def gradient_to_center(h, w):
    return 1.0 - gradient_from_center(h, w)


# === 3. ГОЛОВНИЙ БЛОК ===
def main():
    # Завантаження або генерація зображення
    filename = "images/input.jpg"
    if os.path.exists(filename):
        img = np.array(Image.open(filename).convert('RGB'))
    else:
        print("Файл зображення не знайдено, генеруємо тестове зображення...")
        img = generate_test_image(500, 500)

    h, w, _ = img.shape

    # --- Генерація відфільтрованих версій ---
    img_gray = filter_grayscale(img)
    img_neg = filter_negative(img)
    img_sepia = filter_sepia(img)

    # --- Генерація градієнтних масок ---
    mask_diag = gradient_diagonal(h, w)
    mask_to_center = gradient_to_center(h, w)
    mask_from_center = gradient_from_center(h, w)

    # --- Змішування (Синтез ефектів) ---
    res_1 = apply_gradient_blend(img, img_gray, mask_diag)
    res_2 = apply_gradient_blend(img, img_neg, mask_to_center)
    res_3 = apply_gradient_blend(img, img_sepia, mask_from_center)

    # --- Візуалізація результатів ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.canvas.manager.set_window_title("Рівень III: Матрична обробка зображень")

    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Оригінальне зображення", fontsize=12)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(res_1)
    axs[0, 1].set_title("Відтінки сірого\n(Градієнт: Діагональ)", fontsize=12)
    axs[0, 1].axis('off')

    axs[1, 0].imshow(res_2)
    axs[1, 0].set_title("Негатив\n(Градієнт: До центру)", fontsize=12)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(res_3)
    axs[1, 1].set_title("Сепія\n(Градієнт: Від центру)", fontsize=12)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
