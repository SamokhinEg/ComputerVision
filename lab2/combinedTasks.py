import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ МАТРИЦЬ (2D та 3D)
# ==========================================
def get_2d_translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def get_2d_scaling(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def get_3d_rotation_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


# ==========================================
# ГОЛОВНИЙ СКРИПТ (РІВЕНЬ ІІІ)
# ==========================================
def main():
    # Створення вікна з двома підграфіками: 2D зліва, 3D справа
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title('Рівень III: 2D та 3D перетворення')

    # --- НАЛАШТУВАННЯ 2D (РІВЕНЬ I) ---
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set_title("2D: Квадрат (Переміщення+Масштабування)", fontsize=12)
    ax2d.set_xlim(0, 100)
    ax2d.set_ylim(0, 100)
    ax2d.set_aspect('equal')
    ax2d.grid(True, linestyle='--')

    # Початкова 2D матриця квадрата (Розширена)
    P_2d = np.array([
        [10, 20, 20, 10, 10],
        [10, 10, 20, 20, 10],
        [1, 1, 1, 1, 1]
    ], dtype=float)

    # Виконання 2D перетворень (статично, оскільки траєкторія малюється)
    M1 = get_2d_translation(4, 3) @ get_2d_scaling(1.05, 1.05)
    traj_x, traj_y = [np.mean(P_2d[0, :-1])], [np.mean(P_2d[1, :-1])]

    ax2d.plot(P_2d[0], P_2d[1], color='blue', label='Старт')

    for _ in range(8):
        P_2d = M1 @ P_2d
        traj_x.append(np.mean(P_2d[0, :-1]))
        traj_y.append(np.mean(P_2d[1, :-1]))
        ax2d.plot(P_2d[0], P_2d[1], color='green', alpha=0.3)

    ax2d.plot(traj_x, traj_y, 'o--', color='orange', label='Траєкторія Опер. 1')

    # Операція 2 (Переміщення)
    M2 = get_2d_translation(-20, 15)
    P_final = M2 @ P_2d
    ax2d.plot(P_final[0], P_final[1], color='red', linewidth=2, label='Фініш (Опер. 2)')
    ax2d.plot([traj_x[-1], np.mean(P_final[0, :-1])],
              [traj_y[-1], np.mean(P_final[1, :-1])], ':', color='purple')
    ax2d.legend()

    # --- НАЛАШТУВАННЯ 3D (РІВЕНЬ II) ---
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.set_title("3D: Піраміда (Анімація + Динаміка кольору)", fontsize=12)

    # Встановлення аксонометричної (ортографічної) проекції та ізометричного кута
    ax3d.set_proj_type('ortho')
    ax3d.view_init(elev=35.264, azim=45)

    ax3d.set_xlim(-2, 2)
    ax3d.set_ylim(-2, 2)
    ax3d.set_zlim(0, 3)
    ax3d.set_axis_off()  # Вимикаємо сітку для кращої візуалізації об'єкта

    # Початкова 3D матриця піраміди з 4-кутною основою
    # Вершини: 0..3 - основа (Z=0), 4 - вершина піраміди (Z=2)
    P_3d_base = np.array([
        [-1, 1, 1, -1, 0],  # X
        [-1, -1, 1, 1, 0],  # Y
        [0, 0, 0, 0, 2],  # Z
        [1, 1, 1, 1, 1]  # W (Однорідна координата)
    ], dtype=float)

    # Індекси вершин, що формують грані (1 основа + 4 бічні)
    faces_indices = [
        [0, 1, 2, 3],  # Основа
        [0, 1, 4],  # Бічна 1
        [1, 2, 4],  # Бічна 2
        [2, 3, 4],  # Бічна 3
        [3, 0, 4]  # Бічна 4
    ]

    # Створення об'єкта колекції полігонів для 3D
    poly3d = Poly3DCollection([], alpha=1.0, linewidths=1.5)
    ax3d.add_collection3d(poly3d)

    # --- ФУНКЦІЯ АНІМАЦІЇ ---
    def update(frame):
        # 1. Циклічне обертання: обчислення кута та множення матриць
        theta = np.radians(frame * 2)  # Обертання на 2 градуси кожен кадр
        M_rot = get_3d_rotation_z(theta)

        # Застосування матриці до початкових координат
        P_current = M_rot @ P_3d_base

        # Формування масиву координат для граней
        verts = []
        for face in faces_indices:
            # Витягуємо X, Y, Z для поточних вершин грані
            x_face = P_current[0, face]
            y_face = P_current[1, face]
            z_face = P_current[2, face]
            verts.append(list(zip(x_face, y_face, z_face)))

        poly3d.set_verts(verts)

        # 2. Динаміка: з'являється/гасне (альфа-канал) та змінює колір
        # Використовуємо синусоїду для плавних пульсацій
        alpha_val = abs(np.sin(np.radians(frame * 2.5))) * 0.8 + 0.2  # Змінюється від 0.2 до 1.0

        # Колір змінюється в RGB просторі з часом
        r = abs(np.sin(np.radians(frame * 1.5)))
        g = abs(np.cos(np.radians(frame * 1.0)))
        b = abs(np.sin(np.radians(frame * 0.5)))

        # Встановлюємо динамічні кольори
        poly3d.set_facecolors((r, g, b, alpha_val))

        # Контур змінюється на контрастний
        poly3d.set_edgecolors((1 - r, 1 - g, 1 - b, alpha_val))

        return poly3d,

    # Запуск анімації (interval=50 означає 50 мілісекунд на кадр)
    ani = FuncAnimation(fig, update, frames=360, interval=50, blit=False)

    # Відображення на екрані
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
