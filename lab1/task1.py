import tkinter as tk
import math


def draw_shapes():
    # 1. Ініціалізація вікна та полотна
    root = tk.Tk()
    root.title("Графічні примітиви - Лабораторна робота")

    # Розміри вікна
    canvas_width = 800
    canvas_height = 400
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()

    # Спільні параметри технічного завдання
    line_color = "black"
    color_bg = "white"  # Колір для створення "порожніх" кілець
    color_outer = "#4DA6FF"  # Колір зовнішнього кільця (синій)
    color_middle = "#FF6666"  # Колір середнього кільця (червоний)

    # Масив кольорів для циклів (зовнішнє, порожнє, середнє, внутрішнє порожнє)
    fills = [color_outer, color_bg, color_middle, color_bg]

    # ==========================================
    # ФІГУРА 1: Трикутники (малюється "лініями")
    # ==========================================
    cx_tri, cy_tri = 250, 220  # Центр трикутників
    radii_tri = [160, 115, 70, 25]  # Радіуси від найбільшого до найменшого

    for i in range(4):
        R = radii_tri[i]

        # Обчислення координат вершин за математичними формулами
        x1 = cx_tri
        y1 = cy_tri - R

        x2 = cx_tri - R * math.cos(math.pi / 6)
        y2 = cy_tri + R * math.sin(math.pi / 6)

        x3 = cx_tri + R * math.cos(math.pi / 6)
        y3 = cy_tri + R * math.sin(math.pi / 6)

        # Створення фігури за координатами вершин (лініями, що утворюють полігон)
        canvas.create_polygon(x1, y1, x2, y2, x3, y3,
                              fill=fills[i], outline=line_color, width=2)

    # ==========================================
    # ФІГУРА 2: Квадрати (вбудованими засобами)
    # ==========================================
    cx_sq, cy_sq = 550, 245  # Центр квадратів (трохи зміщений для візуального вирівнювання)
    sizes_sq = [135, 100, 65, 30]  # Половина сторони квадрата

    for i in range(4):
        S = sizes_sq[i]

        # Координати верхнього лівого та нижнього правого кутів
        x_tl = cx_sq - S
        y_tl = cy_sq - S
        x_br = cx_sq + S
        y_br = cy_sq + S

        # Використання спеціалізованого вбудованого методу для прямокутників
        canvas.create_rectangle(x_tl, y_tl, x_br, y_br,
                                fill=fills[i], outline=line_color, width=2)

    # Запуск головного циклу вікна
    root.mainloop()


if __name__ == "__main__":
    draw_shapes()
