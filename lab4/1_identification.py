import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    # 1. Завантаження цифрового зображення
    img = cv2.imread(image_path)
    if img is None:
        print("Помилка: Зображення не знайдено!")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Для коректного відображення в Matplotlib

    # 2. Корекція кольору (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Корекція гістограми (Локальна - CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Фільтрація (Двосторонній фільтр для збереження меж)
    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Векторизація (Пошук меж алгоритмом Canny)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Морфологічне закриття, щоб з'єднати розірвані контури стін
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Пошук векторних контурів
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Ідентифікація об'єкта за геометричною ознакою
    result_img = img_rgb.copy()
    buildings_found = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Відсіюємо занадто дрібні (шум) та занадто великі об'єкти (вся вулиця)
        if 1000 < area < 100000:
            # Апроксимація контуру до багатокутника (геометрична ознака)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # Якщо фігура має від 4 до 10 кутів (типова форма стін/дахів будинків)
            if 4 <= len(approx) <= 10:
                buildings_found += 1
                # Малюємо контур (зелений колір)
                cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 3)

                # Додаємо підпис
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result_img, "Building", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Візуалізація R&D етапів
    titles = ['Оригінал', 'CLAHE + Bilateral Filter', 'Векторизація (Межі)',
              f'Ідентифікація ({buildings_found} знайдено)']
    images = [img_rgb, blurred, closed_edges, result_img]

    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Виклик функції (замініть на шлях до вашого фото)
process_image('crossroad.jpg')
