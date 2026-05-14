import cv2
import sys


def get_tracker():
    """Безпечна ініціалізація трекера для різних версій OpenCV"""
    try:
        # Для старіших версій OpenCV 3.x та деяких 4.x
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            # Для нових версій OpenCV 4.5+ (трекери перенесли в legacy)
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            print("\nКРИТИЧНА ПОМИЛКА: Модуль трекінгу не знайдено!")
            print("Виконайте в терміналі команду:")
            print("pip install opencv-contrib-python")
            sys.exit()


def track_object(video_path):
    print("--- Запуск Object Tracking (CSRT) ---")

    # 1. Створюємо трекер
    tracker = get_tracker()

    # 2. Відкриваємо відео. Якщо передати 0, увімкнеться веб-камера
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Помилка: Не вдалося відкрити джерело відео ({video_path})")
        return

    # Зчитуємо перший кадр для виділення об'єкта
    success, frame = cap.read()
    if not success:
        print("Не вдалося прочитати перший кадр.")
        return

    # Для великих відео трохи зменшуємо кадр, щоб вікно помістилося на екрані
    frame = cv2.resize(frame, (800, 600))

    # 3. Вибір об'єкта мишкою
    print("\nІНСТРУКЦІЯ:")
    print("1. Намалюйте мишкою прямокутник навколо об'єкта.")
    print("2. Натисніть клавішу ENTER або ПРОБІЛ, щоб почати стеження.")
    print("3. Натисніть клавішу 'q', щоб зупинити скрипт.")

    # Спеціальне вікно OpenCV для вибору області
    bbox = cv2.selectROI("Виділіть об'єкт та натисніть ENTER", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Виділіть об'єкт та натисніть ENTER")

    # Перевірка, чи користувач дійсно щось виділив (а не просто натиснув Enter)
    if bbox[2] == 0 or bbox[3] == 0:
        print("Об'єкт не виділено. Вихід...")
        return

    # Ініціалізуємо трекер першим кадром та координатами рамки
    tracker.init(frame, bbox)

    while True:
        # Читаємо наступний кадр
        success, frame = cap.read()
        if not success:
            print("Відео завершилося.")
            break

        frame = cv2.resize(frame, (800, 600))

        # 4. Оновлюємо трекер (він шукає об'єкт на новому кадрі)
        success, bbox = tracker.update(frame)

        if success:
            # Об'єкт знайдено! Розпаковуємо координати та малюємо зелену рамку
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "CSRT Tracker: TRACKING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Об'єкт втрачено (перекрили, вийшов за кадр) - малюємо червоний текст
            cv2.putText(frame, "CSRT Tracker: LOST", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 5. Показуємо результат
        cv2.imshow("Object Tracking", frame)

        # Вихід за натисканням клавіші 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Зупинено користувачем.")
            break

    # Звільняємо ресурси
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    SOURCE = "crossroad.mp4"
    track_object(SOURCE)
