import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def process_and_match(img_operative_path, img_highres_path):
    # 1. Завантаження зображень
    img1 = cv2.imread(img_operative_path, cv2.IMREAD_GRAYSCALE)  # Низька роздільна здатність
    img2 = cv2.imread(img_highres_path, cv2.IMREAD_GRAYSCALE)  # Висока роздільна здатність

    if img1 is None or img2 is None:
        print("Помилка завантаження зображень!")
        return

    # 2. Фільтрація (імітація п.3-5 з попередньої ЛР) - Bilateral Filter
    img1_filtered = cv2.bilateralFilter(img1, 9, 75, 75)
    img2_filtered = cv2.bilateralFilter(img2, 9, 75, 75)

    # 3. Ініціалізація дескриптора ORB
    orb = cv2.ORB_create(nfeatures=500)

    # 4. Знаходження ключових точок та дескрипторів
    kp1, des1 = orb.detectAndCompute(img1_filtered, None)
    kp2, des2 = orb.detectAndCompute(img2_filtered, None)

    # 5. Порівняння дескрипторів за допомогою Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Сортування збігів за відстанню (кращі збіги на початку)
    matches = sorted(matches, key=lambda x: x.distance)

    # Беремо топ-50 найкращих збігів для розрахунку
    good_matches = matches[:50]
    num_matches = len(good_matches)

    # 6. Розрахунок ймовірності ідентифікації
    # Припустимо, що 30 якісних збігів - це 100% впевненість для нашого типу об'єкта (резервуара)
    THRESHOLD_MATCHES = 30
    probability = min(100.0, (num_matches / THRESHOLD_MATCHES) * 100.0)

    print(f"Кількість знайдених збігів особливих точок: {num_matches}")
    print(f"Ймовірність ідентифікації об'єкта: {probability:.2f}%")

    # 7. Візуалізація результатів
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Зіставлення дескрипторів ORB (Ймовірність: {probability:.2f}%)")
    plt.axis('off')
    plt.show()

# Виклик функції (підставте свої шляхи до фото резервуарів)
process_and_match('images/operative.jpg', 'images/high_res.jpg')
