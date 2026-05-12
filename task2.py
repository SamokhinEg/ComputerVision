import tkinter as tk


def draw_logo_lines(canvas, cx, cy, R, W, colors):
    """Малювання логотипу за допомогою ліній"""
    # Верхній трикутник
    canvas.create_line(cx, cy, cx - W / 2, cy - R, fill=colors[0], width=3)
    canvas.create_line(cx - W / 2, cy - R, cx + W / 2, cy - R, fill=colors[0], width=3)
    canvas.create_line(cx + W / 2, cy - R, cx, cy, fill=colors[0], width=3)

    # Правий трикутник
    canvas.create_line(cx, cy, cx + R, cy - W / 2, fill=colors[1], width=3)
    canvas.create_line(cx + R, cy - W / 2, cx + R, cy + W / 2, fill=colors[1], width=3)
    canvas.create_line(cx + R, cy + W / 2, cx, cy, fill=colors[1], width=3)

    # Нижній трикутник
    canvas.create_line(cx, cy, cx + W / 2, cy + R, fill=colors[2], width=3)
    canvas.create_line(cx + W / 2, cy + R, cx - W / 2, cy + R, fill=colors[2], width=3)
    canvas.create_line(cx - W / 2, cy + R, cx, cy, fill=colors[2], width=3)

    # Лівий трикутник
    canvas.create_line(cx, cy, cx - R, cy + W / 2, fill=colors[3], width=3)
    canvas.create_line(cx - R, cy + W / 2, cx - R, cy - W / 2, fill=colors[3], width=3)
    canvas.create_line(cx - R, cy - W / 2, cx, cy, fill=colors[3], width=3)


def draw_logo_polygons(canvas, cx, cy, R, W, fills, outlines):
    """Малювання логотипу за допомогою багатокутників (із заливкою)"""
    # Верхній трикутник
    canvas.create_polygon(cx, cy, cx - W / 2, cy - R, cx + W / 2, cy - R,
                          fill=fills[0], outline=outlines[0], width=3)
    # Правий трикутник
    canvas.create_polygon(cx, cy, cx + R, cy - W / 2, cx + R, cy + W / 2,
                          fill=fills[1], outline=outlines[1], width=3)
    # Нижній трикутник
    canvas.create_polygon(cx, cy, cx + W / 2, cy + R, cx - W / 2, cy + R,
                          fill=fills[2], outline=outlines[2], width=3)
    # Лівий трикутник
    canvas.create_polygon(cx, cy, cx - R, cy + W / 2, cx - R, cy - W / 2,
                          fill=fills[3], outline=outlines[3], width=3)


def main():
    root = tk.Tk()
    root.title("Малий логотип компанії")

    canvas = tk.Canvas(root, width=800, height=800, bg='white')
    canvas.pack()

    R = 100  # Висота трикутників
    W = 100  # Ширина зовнішньої основи

    # --- Колірні схеми ---
    # 1. Монохромна (чорний колір для ліній, сірий для заливки)
    mono_lines = ["black", "black", "black", "black"]
    mono_fills = ["#D3D3D3", "#A9A9A9", "#D3D3D3", "#A9A9A9"]  # Світло-сірий та темно-сірий

    # 2. Кольорова (Тетрадична схема: Червоний, Синій, Жовтий, Зелений)
    color_lines = ["#E63946", "#1D3557", "#F4A261", "#2A9D8F"]
    color_fills = ["#FF9999", "#A8DADC", "#FFE8A1", "#81B29A"]  # Світліші відтінки для заливки
    color_outlines = ["#990000", "#001D3D", "#CC5500", "#1B4332"]  # Темні відтінки для граней

    # --- 1. Лініями (Монохромний) ---
    draw_logo_lines(canvas, 200, 200, R, W, mono_lines)
    canvas.create_text(200, 340, text="1. Примітив 'Лінія' (Монохром)", font=("Arial", 14, "bold"))

    # --- 2. Лініями (Кольоровий) ---
    draw_logo_lines(canvas, 600, 200, R, W, color_lines)
    canvas.create_text(600, 340, text="2. Примітив 'Лінія' (Колір)", font=("Arial", 14, "bold"))

    # --- 3. Багатокутниками (Монохромний) ---
    draw_logo_polygons(canvas, 200, 600, R, W, mono_fills, mono_lines)
    canvas.create_text(200, 740, text="3. Багатокутник (Монохром)", font=("Arial", 14, "bold"))

    # --- 4. Багатокутниками (Кольоровий із заливкою) ---
    draw_logo_polygons(canvas, 600, 600, R, W, color_fills, color_outlines)
    canvas.create_text(600, 740, text="4. Багатокутник (Колір+Заливка)", font=("Arial", 14, "bold"))

    root.mainloop()


if __name__ == "__main__":
    main()