import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def process_satellite_image(image_path, title):
    print(f"\n--- Обробка: {title} ---")

    # 2. Отримання цифрового растрового знімка
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Файл {image_path} не знайдено!")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Кольорова корекція та фільтрація (Bilateral Filter)
    # Зберігає різкість меж берегової лінії, усуваючи шум хвиль/пікселізації
    filtered_img = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Кольорова кластеризація (K-Means)
    # Перетворюємо зображення у 2D масив пікселів
    pixels = filtered_img.reshape((-1, 3))

    # Використовуємо K=3 (напр. Вода, Ліс, Земля)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Знаходимо кластер води. На супутникових знімках вода зазвичай найтемніша.
    # Обчислюємо середню яскравість (суму RGB) для кожного центру кластера
    cluster_centers = kmeans.cluster_centers_
    darkest_cluster_idx = np.argmin(np.sum(cluster_centers, axis=1))

    # 5. Сегментація кластеризованого зображення
    # Створюємо бінарну маску для кластера "Вода"
    water_mask = np.uint8(labels == darkest_cluster_idx).reshape(img_rgb.shape[:2]) * 255

    # Морфологічні операції: видаляємо шум (острівці) та заповнюємо дірки (човни/бліки)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    # Виділення контурів
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Знаходимо найбільший контур (основна водойма)
    if not contours:
        print("Водойму не знайдено!")
        return img_rgb, water_mask, None

    main_contour = max(contours, key=cv2.contourArea)

    # Малюємо знайдений контур на копії оригінального зображення
    result_img = img_rgb.copy()
    cv2.drawContours(result_img, [main_contour], -1, (255, 0, 0), 3)  # Червоний контур

    return result_img, water_mask, main_contour


def main():
    # Шляхи до файлів (переконайтеся, що вони існують у папці)
    operative_path = 'images/operative.jpg'  # Низька роздільна здатність (Landsat)
    highres_path = 'images/high_res.jpg'  # Висока роздільна здатність (Bing)

    try:
        # Обробка обох знімків
        res_op, mask_op, contour_op = process_satellite_image(operative_path, "Оперативні дані (Landsat)")
        res_hr, mask_hr, contour_hr = process_satellite_image(highres_path, "Високоточні дані (Bing Maps)")

        # 6. Ідентифікація (Програмне порівняння контурів)
        print("\n--- 6. Ідентифікація та порівняння ---")
        if contour_op is not None and contour_hr is not None:
            # Використання Моментів Ху для порівняння форми (інваріантно до масштабу)
            # Чим ближче значення до 0.0, тим більше контури схожі за геометрією
            similarity = cv2.matchShapes(contour_op, contour_hr, cv2.CONTOURS_MATCH_I1, 0.0)

            print(f"Коефіцієнт відмінності геометрії (Hu Moments): {similarity:.4f}")
            if similarity < 0.2:  # Емпіричний поріг для знімків різної якості
                print("-> РЕЗУЛЬТАТ: Контури збігаються. Об'єкт успішно ідентифіковано як 'Водойма'.")
            else:
                print("-> РЕЗУЛЬТАТ: Геометрія суттєво відрізняється. Можливі сезонні зміни або помилка сегментації.")

        # Візуалізація результатів
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Розрізнення та ідентифікація об’єктів за даними ДЗЗ (Водойми)', fontsize=16)

        # Верхній ряд: Оперативні дані
        axs[0, 0].imshow(cv2.cvtColor(cv2.imread(operative_path), cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Оригінал (Оперативні / Низька якість)')
        axs[0, 1].imshow(mask_op, cmap='gray')
        axs[0, 1].set_title('Кластеризація (Маска води)')
        axs[0, 2].imshow(res_op)
        axs[0, 2].set_title('Векторизація контуру')

        # Нижній ряд: Високоточні дані
        axs[1, 0].imshow(cv2.cvtColor(cv2.imread(highres_path), cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Оригінал (Еталонні / Висока якість)')
        axs[1, 1].imshow(mask_hr, cmap='gray')
        axs[1, 1].set_title('Кластеризація (Маска води)')
        axs[1, 2].imshow(res_hr)
        axs[1, 2].set_title('Векторизація контуру')

        for ax in axs.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Сталася помилка: {e}")


if __name__ == "__main__":
    main()
