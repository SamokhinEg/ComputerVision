import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Розмір графічного вікна (растрового буфера)
WIDTH, HEIGHT = 400, 400

# ==========================================
# 1. РАСТРОВІ АЛГОРИТМИ (ЗАВДАННЯ I та II)
# ==========================================

def draw_line_bresenham(img, x0, y0, x1, y1):
    """Алгоритм Брезенхема для растеризації ліній (Завдання I).
       Зміна кольору контуру від растра до растра (по осі Y)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < WIDTH and 0 <= y0 < HEIGHT:
            # Зміна кольору лінії залежно від Y-координати (растра)
            # Градієнт від червоного до жовтого
            factor = y0 / HEIGHT
            img[y0, x0] = [255, int(255 * factor), 0]
            
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def fill_polygon_scanline(img, pts, base_color):
    """Построковий алгоритм заливки багатокутника (Завдання II).
       Зміна кольору заливки від растра до растра (градієнт по Y)."""
    min_y = max(0, int(np.min(pts[:, 1])))
    max_y = min(HEIGHT - 1, int(np.max(pts[:, 1])))

    # Проходимо по кожному растровому рядку (scanline)
    for y in range(min_y, max_y + 1):
        intersections = []
        # Знаходимо перетин поточного рядка Y з ребрами багатокутника
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
            # Якщо рядок перетинає ребро (не включаючи верхню вершину для уникнення дублікатів)
            if y_min <= y < y_max:
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections.append(int(x))
                
        intersections.sort()
        
        # Зафарбовуємо відрізки між парами точок перетину
        for i in range(0, len(intersections) - 1, 2):
            x_start = max(0, intersections[i])
            x_end = min(WIDTH - 1, intersections[i+1])
            
            # Зміна кольору заливки грані від растра до растра
            factor = y / HEIGHT
            r = int(base_color[0] * (1 - factor))
            g = int(base_color[1] * factor)
            b = base_color[2]
            
            # Малюємо горизонтальну лінію растра
            img[y, x_start:x_end+1] = [r, g, b]

# ==========================================
# 2. МАТЕМАТИЧНІ МОДЕЛІ 3D (АКСОНОМЕТРІЯ ТА ОБЕРТАННЯ)
# ==========================================

def get_rotation_matrix(angle_x, angle_y):
    """Матриця обертання навколо осей X та Y"""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    
    return np.dot(Ry, Rx)

def is_front_facing(pts):
    """Відсікання невидимих граней (Backface Culling) 
       за допомогою векторного добутку 2D-координат."""
    p1, p2, p3 = pts[0], pts[1], pts[2]
    cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
    return cross_product > 0

# ==========================================
# 3. ГОЛОВНИЙ ЦИКЛ РЕНДЕРУ (АНІМАЦІЯ)
# ==========================================

def main():
    # Вершини куба від -1 до 1
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]
    ])
    
    # Індекси вершин для 6 граней куба (послідовність за годинниковою стрілкою)
    faces = [
        [0, 1, 2, 3], # Передня
        [5, 4, 7, 6], # Задня
        [4, 0, 3, 7], # Ліва
        [1, 5, 6, 2], # Права
        [4, 5, 1, 0], # Нижня
        [3, 2, 6, 7]  # Верхня
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title('Рівні I та II: Програмна растеризація')
    ax.axis('off')
    
    # Ініціалізація порожнього зображення (RGB буфер)
    img_display = ax.imshow(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    
    def update(frame):
        # Очищення растрового буфера (чорний фон)
        frame_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Кути обертання для поточної секунди
        angle_x = np.radians(frame * 2)
        angle_y = np.radians(frame * 3)
        R = get_rotation_matrix(angle_x, angle_y)
        
        # Аксонометрична проекція: Обертаємо вершини
        rotated_vertices = np.dot(vertices, R.T)
        
        # Проекція на площину 2D (ортографічна) з масштабуванням у центр екрану
        scale = 100
        projected_2d = np.zeros((8, 2))
        for i in range(8):
            projected_2d[i][0] = int(rotated_vertices[i][0] * scale + WIDTH / 2)
            projected_2d[i][1] = int(rotated_vertices[i][1] * scale + HEIGHT / 2)
            
        # Малювання фігури
        for face in faces:
            pts_2d = projected_2d[face]
            
            # Рендеримо тільки грані, які "дивляться" на нас
            if is_front_facing(pts_2d):
                # Завдання II: Заливка грані зі зміною кольору від растра до растра
                base_color = [50, 150, 255] # Базовий синій для граней
                fill_polygon_scanline(frame_buffer, pts_2d, base_color)
                
                # Завдання I: Малювання контурних ліній зі зміною кольору від растра до растра
                for i in range(len(face)):
                    p1 = pts_2d[i]
                    p2 = pts_2d[(i + 1) % len(face)]
                    draw_line_bresenham(frame_buffer, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # Оновлення зображення на екрані
        img_display.set_data(frame_buffer)
        return [img_display]

    # Запуск циклу анімації
    ani = animation.FuncAnimation(fig, update, frames=360, interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
