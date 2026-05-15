import cv2
import numpy as np
import matplotlib.pyplot as plt


def stereo_reconstruction(left_img_path, right_img_path):
    print("--- Запуск 3D реконструкції зі стереопари ---")

    # 1. Завантаження зображень (у відтінках сірого для алгоритму)
    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Помилка: Неможливо завантажити зображення. Перевірте шляхи.")
        return

    # 2. Налаштування алгоритму StereoSGBM (Semi-Global Block Matching)
    # Ці параметри залежать від вашої бази камер та роздільної здатності
    window_size = 5
    min_disp = 0
    num_disp = 16 * 5  # Має бути кратно 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 3. Обчислення карти зміщення (Disparity Map)
    print("Обчислення карти зміщення...")
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # 4. Тривимірна реконструкція (Reprojection)
    print("Генерація 3D хмари точок...")
    h, w = imgL.shape
    # Матриця Q - матриця перспективного перетворення (зазвичай отримується після калібрування)
    # Тут ми використовуємо синтетичну матрицю для демонстрації
    focal_length = 0.8 * w  # Приблизна фокусна відстань
    Q = np.float32([
        [1, 0, 0, -0.5 * w],
        [0, -1, 0, 0.5 * h],  # -1 для правильної орієнтації Y
        [0, 0, 0, -focal_length],
        [0, 0, 1, 0]
    ])

    # Конвертація 2D + Disparity -> 3D координати
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # 5. Візуалізація результатів
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(imgL, cmap='gray')
    plt.title('Лівий канал')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imgR, cmap='gray')
    plt.title('Правий канал')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Нормалізація для відображення
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.imshow(disp_vis, cmap='plasma')
    plt.title('Карта глибини (Disparity)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Опціонально: збереження хмари точок у форматі .ply для перегляду в MeshLab або Blender
    # save_point_cloud(points_3D, imgL)


if __name__ == "__main__":
    # Сфотографуйте об'єкт, зсуньте камеру на 5-10 см вправо і зробіть ще одне фото
    stereo_reconstruction('images/right.jpg', 'images/left.jpg')
