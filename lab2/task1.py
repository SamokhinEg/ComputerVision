import numpy as np
import matplotlib.pyplot as plt


# --- 1. Функції для створення матриць перетворень ---
def get_translation_matrix(tx, ty):
    """Повертає матрицю переміщення 3x3"""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def get_scaling_matrix(sx, sy):
    """Повертає матрицю масштабування 3x3"""
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def get_center(P):
    """Обчислює геометричний центр фігури за її матрицею координат"""
    # Ігноруємо останній стовпець (замикання), щоб не дублювати 1-шу точку
    x_c = np.mean(P[0, :-1])
    y_c = np.mean(P[1, :-1])
    return x_c, y_c


def main():
    # --- 2. Ініціалізація розширеної матриці квадрата ---
    # Квадрат 10x10. Вершини з'єднуються по колу (остання точка = першій)
    # Формат рядків: X, Y, 1 (однорідні координати)
    P = np.array([
        [10, 20, 20, 10, 10],  # X
        [10, 10, 20, 20, 10],  # Y
        [1, 1, 1, 1, 1]  # 1
    ], dtype=float)

    # Налаштування вікна
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("2D Матричні перетворення", fontsize=14, pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')  # Пропорції 1:1, щоб квадрат не спотворювався

    # Відображення початкового стану
    ax.plot(P[0, :], P[1, :], color='blue', linewidth=2, label='Початковий стан')

    # --- 3. Операція 1 (Переміщення + Масштабування циклічно) ---
    # Параметри 1-го перетворення
    sx, sy = 1.05, 1.05  # Збільшення на 5% кожного кроку
    tx, ty = 4, 3  # Зсув по X та Y

    # Композиційна матриця (Масштабування -> Переміщення)
    M1 = get_translation_matrix(tx, ty) @ get_scaling_matrix(sx, sy)

    steps = 8
    trajectory_x = [get_center(P)[0]]
    trajectory_y = [get_center(P)[1]]

    # Виконання циклу
    for i in range(steps):
        # Матричне множення M1 на матрицю точок P
        P = M1 @ P

        # Збереження центрів для траєкторії
        cx, cy = get_center(P)
        trajectory_x.append(cx)
        trajectory_y.append(cy)

        # Малювання проміжних кроків
        ax.plot(P[0, :], P[1, :], color='green', alpha=0.3)

    # Малюємо траєкторію
    ax.plot(trajectory_x, trajectory_y, color='orange', linestyle='--', marker='o',
            markersize=4, label='Траєкторія Опер. 1')

    # --- 4. Операція 2 (Тільки переміщення) ---
    # Від кінцевої точки Операції 1 зміщуємось в інший бік
    tx2, ty2 = -25, 15
    M2 = get_translation_matrix(tx2, ty2)

    # Застосування фінальної матриці
    P_final = M2 @ P

    # Малювання фінального стану
    ax.plot(P_final[0, :], P_final[1, :], color='red', linewidth=3,
            label='Фінальний стан (Після Опер. 2)')

    # З'єднання траєкторією між кінцем Опер 1 та результатом Опер 2
    cx_final, cy_final = get_center(P_final)
    ax.plot([trajectory_x[-1], cx_final], [trajectory_y[-1], cy_final],
            color='purple', linestyle=':', linewidth=2, label='Вектор Опер. 2 (Переміщення)')

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
