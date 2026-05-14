import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Імпортуємо PyTorch замість TensorFlow
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def extract_features_pytorch(image_paths):
    # Завантаження попередньо навченої моделі MobileNetV2 з PyTorch
    # weights='DEFAULT' означає використання найкращих доступних ваг (ImageNet)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Видаляємо останній шар (класифікатор), залишаємо тільки шар ознак
    model.classifier = torch.nn.Identity()
    model.eval()  # Переводимо модель у режим оцінки

    # Стандартні трансформації зображень для моделей PyTorch
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []

    # Вимикаємо розрахунок градієнтів для прискорення роботи
    with torch.no_grad():
        for path in image_paths:
            try:
                # Завантажуємо зображення через PIL (стандарт для PyTorch)
                img = Image.open(path).convert('RGB')
                input_tensor = preprocess(img)
                input_batch = input_tensor.unsqueeze(0)  # Додаємо розмірність батчу

                # Отримання вектору ознак
                feat = model(input_batch)
                features.append(feat.numpy().flatten())
            except Exception as e:
                print(f"Помилка обробки файлу {path}: {e}")

    return np.array(features)


def cluster_dataset(dataset_folder, num_clusters=2):
    # Збираємо всі файли зображень з папки
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print(f"Помилка: У папці '{dataset_folder}' не знайдено зображень!")
        return

    print(f"Знайдено зображень: {len(image_files)}. Витягування ознак через PyTorch...")
    features = extract_features_pytorch(image_files)

    if len(features) == 0:
        print("Не вдалося витягти ознаки. Перевірте файли.")
        return

    print(f"Кластеризація на {num_clusters} групи (K-Means)...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Виведення результатів кластеризації
    for cluster_id in range(num_clusters):
        print(f"\n--- Кластер {cluster_id} ---")
        cluster_images = [img for i, img in enumerate(image_files) if labels[i] == cluster_id]

        if not cluster_images:
            continue

        # Візуалізація перших 5 зображень з кластера
        num_to_show = min(5, len(cluster_images))
        fig, axes = plt.subplots(1, num_to_show, figsize=(15, 3))

        # Якщо зображення лише одне, axes не буде масивом, тому робимо його масивом
        if num_to_show == 1:
            axes = [axes]

        for ax, img_path in zip(axes, cluster_images[:5]):
            img = cv2.imread(img_path)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(os.path.basename(img_path)[:15] + "...")  # Скорочуємо довгі назви
            ax.axis('off')

        plt.suptitle(f"Зображення з Кластеру {cluster_id}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # ВКАЖІТЬ ТУТ ШЛЯХ ДО ВАШОЇ ПАПКИ З КАРТИНКАМИ
    folder_path = 'images'

    # Перевіряємо, чи існує папка, якщо ні - створюємо її, щоб скрипт не падав
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Створено папку '{folder_path}'. Покладіть туди кілька фото та запустіть скрипт ще раз.")
    else:
        # Запускаємо кластеризацію (спробуємо поділити на 2 змістовні групи)
        cluster_dataset(folder_path, num_clusters=2)
