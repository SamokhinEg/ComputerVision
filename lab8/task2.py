import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_and_match(img1, img2):
    """Знаходження та співставлення особливих точок між двома камерами"""
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Захист від порожніх зображень (якщо SIFT нічого не знайшов)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([]), np.array([])

    # FLANN алгоритм
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Фільтр Лоу (Lowe's ratio test)
    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.float32(pts1), np.float32(pts2)


def reconstruct_3d_from_views(image_paths):
    print("--- Запуск Багатовидової 3D Реконструкції (SfM) ---")

    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Помилка: Неможливо завантажити '{path}'. Перевірте чи існує файл.")
            return
        images.append(img)

    if len(images) < 2:
        print("Помилка: Потрібно мінімум 2 зображення!")
        return

    # Приблизна матриця внутрішніх параметрів камери
    h, w = images[0].shape
    focal_length = w
    K = np.array([[focal_length, 0, w // 2],
                  [0, focal_length, h // 2],
                  [0, 0, 1]], dtype=np.float64)

    all_3d_points = []

    # Матриця проекції першої камери (вважаємо її початком координат)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = np.dot(K, P1)

    for i in range(len(images) - 1):
        print(f"\nОбробка пари камер: {i + 1} та {i + 2}...")

        pts1, pts2 = extract_and_match(images[i], images[i + 1])
        print(f"Знайдено {len(pts1)} спільних точок.")

        if len(pts1) < 10:
            print("Замало спільних точок для реконструкції. Пропускаємо пару.")
            continue

        # Знаходження Ессенціальної матриці
        E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or E.shape != (3, 3):
            print("Помилка обчислення матриці E. Пропускаємо пару.")
            continue

        # Відновлення положення другої камери
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Матриця проекції другої камери
        P2 = np.hstack((R, t))
        P2 = np.dot(K, P2)

        # Відсіюємо неправильні точки (де маска == 255)
        valid_mask = mask_pose.ravel() > 0
        pts1_valid = pts1[valid_mask]
        pts2_valid = pts2[valid_mask]

        if len(pts1_valid) == 0:
            print("Немає коректних точок після фільтрації RANSAC.")
            continue

        # ГАРАНТІЯ БЕЗПЕКИ ДЛЯ OPENCV C++:
        # 1. Приводимо до типу float64
        # 2. Робимо пам'ять безперервною (ascontiguousarray)
        # 3. Транспонуємо масиви точок у формат 2xN (T)
        P1_c = np.ascontiguousarray(P1, dtype=np.float64)
        P2_c = np.ascontiguousarray(P2, dtype=np.float64)
        pts1_c = np.ascontiguousarray(pts1_valid.T, dtype=np.float64)
        pts2_c = np.ascontiguousarray(pts2_valid.T, dtype=np.float64)

        try:
            # ТРІАНГУЛЯЦІЯ
            points_4d_hom = cv2.triangulatePoints(P1_c, P2_c, pts1_c, pts2_c)

            # Переведення з однорідних координат у 3D
            points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
            all_3d_points.append(points_3d.T)
            print(f"Успішно тріангульовано точок: {points_3d.shape[1]}")

        except Exception as e:
            print(f"Помилка тріангуляції: {e}")

    if not all_3d_points:
        print(
            "\nКРИТИЧНО: Не вдалося відновити жодної 3D точки. Спробуйте інші фотографії (об'єкт має бути чітким, ракурс змінений не більше ніж на 10-15 градусів).")
        return

    # Об'єднання точок з усіх пар камер у єдину хмару
    point_cloud = np.vstack(all_3d_points)

    # ВІЗУАЛІЗАЦІЯ В MATPLOTLIB
    print(f"\nГенерація 3D візуалізації ({len(point_cloud)} загальних точок)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Фільтрація екстремальних викидів (шум)
    z_vals = point_cloud[:, 2]
    median_z = np.median(z_vals)
    std_z = np.std(z_vals)
    valid_mask = np.abs(z_vals - median_z) < 2 * std_z
    filtered_points = point_cloud[valid_mask]

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]
    z = filtered_points[:, 2]

    # Малюємо точки
    scatter = ax.scatter(x, z, -y, c=z, cmap='viridis', marker='.', s=15)

    ax.set_title("Мультиканальна 3D Реконструкція (SfM)")
    ax.set_xlabel("X (Ширина)")
    ax.set_ylabel("Z (Глибина)")
    ax.set_zlabel("Y (Висота)")
    plt.colorbar(scatter, label="Відстань до камери")
    plt.show()


if __name__ == "__main__":
    # Вкажіть шляхи до ваших фотографій.
    # Бажано 3 фотографії одного об'єкта, де камера трохи зміщується вбік
    camera_views = ['images/right.jpg', 'images/left.jpg', 'images/top.jpg']

    reconstruct_3d_from_views(camera_views)
