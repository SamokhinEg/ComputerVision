import numpy as np
import matplotlib.pyplot as plt


def main():
    # Задаємо номер за списком (можете змінити на свій)
    a = 15
    k = a * 0.01

    # Задаємо діапазон по X.
    # Використовуємо [0, 1], щоб центр (0,0) був саме на лівій межі,
    # а також тому, що arccos та arcsin існують лише в межах [-1, 1].
    x = np.linspace(0, 1, 200)

    # Математичні моделі сигналів
    y1 = k * np.arccos(x)
    y2 = k * np.arctan(x)
    y3 = k * np.arcsin(x)

    # Максимальне значення для осі Y, щоб відцентрувати вісь X по вертикалі
    y_max = k * (np.pi / 2) * 1.2  # Додаємо 20% відступу для краси графіка

    # Створення графічного вікна 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.canvas.manager.set_window_title('Епюри тестових сигналів')
    fig.suptitle(f"Аналіз тестових сигналів підсилювача (Варіант a={a})", fontsize=16)

    def setup_axes(ax, title):
        """Функція для налаштування зовнішнього вигляду осей"""
        ax.set_title(title, pad=20)

        # Налаштування меж для центрування осі X та розміщення осі Y зліва
        ax.set_xlim(0, 1.1)
        ax.set_ylim(-y_max, y_max)

        # Переносимо осі так, щоб початок координат (0,0) був по центру лівої межі
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Додаємо сітку та підписи
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlabel("Час, x", loc='right')
        ax.set_ylabel("Амплітуда, y(x)", loc='top')

    # 1. Графік першого сигналу
    axs[0, 0].plot(x, y1, color='blue', linewidth=2, label='Закон I: y = k·arccos(x)')
    setup_axes(axs[0, 0], 'Епюр сигналу №1')
    axs[0, 0].legend(loc='upper right')

    # 2. Графік другого сигналу
    axs[0, 1].plot(x, y2, color='green', linewidth=2, label='Закон II: y = k·arctan(x)')
    setup_axes(axs[0, 1], 'Епюр сигналу №2')
    axs[0, 1].legend(loc='upper right')

    # 3. Графік третього сигналу
    axs[1, 0].plot(x, y3, color='red', linewidth=2, label='Закон III: y = k·arcsin(x)')
    setup_axes(axs[1, 0], 'Епюр сигналу №3')
    axs[1, 0].legend(loc='lower right')

    # 4. Графік усіх сигналів сумісно
    axs[1, 1].plot(x, y1, color='blue', linewidth=2, label='Сигнал 1 (arccos)')
    axs[1, 1].plot(x, y2, color='green', linewidth=2, label='Сигнал 2 (arctan)')
    axs[1, 1].plot(x, y3, color='red', linewidth=2, label='Сигнал 3 (arcsin)')
    setup_axes(axs[1, 1], 'Сумісне відображення епюрів')
    axs[1, 1].legend(loc='right')

    # Коригування відступів між графіками
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Залишаємо місце для головного заголовка

    # Відображення графіків
    plt.show()


if __name__ == "__main__":
    main()