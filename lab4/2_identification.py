import cv2
import numpy as np


def process_video(video_path):
    # Ініціалізація захоплення відео
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Помилка: Неможливо відкрити відеопотік!")
        return

    # Створюємо об'єкт CLAHE заздалегідь для оптимізації
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    print("Натисніть 'q' для виходу з вікна відео.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Відео завершено або перервано.")
            break

        # Зменшення розміру кадру для прискорення обробки в реальному часі (опціонально)
        frame = cv2.resize(frame, (800, 600))

        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Локальна корекція гістограми (CLAHE)
        enhanced = clahe.apply(gray)

        # 3. Bilateral Filter
        blurred = cv2.bilateralFilter(enhanced, 7, 50, 50)

        # 4. Векторизація (Canny)
        edges = cv2.Canny(blurred, 50, 150)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Пошук контурів
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Ідентифікація
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Параметри площі можуть залежати від роздільної здатності вашого відео
            if 1500 < area < 80000:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

                # Геометрична ознака будинку (багатокутник)
                if 4 <= len(approx) <= 10:
                    # Малюємо контур
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                    # Визначаємо центр мас для підпису
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(frame, "Bldg", (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Відображення результату
        cv2.imshow('Computer Vision: Building Identification (Real-Time)', frame)

        # Відображення вікна з маскою (Edges) для відлагодження
        cv2.imshow('Vectorization (Edges)', closed_edges)

        # Вихід по натисканню клавіші 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Виклик функції (вставте шлях до відеофайлу перехрестя, або 0 для веб-камери)
process_video('crossroad.mp4')
