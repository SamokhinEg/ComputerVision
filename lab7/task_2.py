import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import Image

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================
# Вкажіть шлях до вашої папки з картинками.
# Структура має бути такою: dataset_path/Клас_1/фото.jpg, dataset_path/Клас_2/фото.jpg
DATASET_PATH = 'images'  # Створіть цю папку поруч зі скриптом


def prepare_data(data_dir, batch_size=4):
    # Трансформації: масштабування до 224x224, перетворення в тензор та нормалізація
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Завантаження датасету
    dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Створення завантажувача даних (DataLoader)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_names = dataset.classes

    return dataloader, class_names


# ==========================================
# 2. КОНСТРУЮВАННЯ НЕЙРОМЕРЕЖІ (Transfer Learning)
# ==========================================
def build_model(num_classes):
    print("Конструювання нейромережі (завантаження MobileNetV2)...")
    # Завантажуємо попередньо навчену архітектуру
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Заморожуємо базові шари, щоб вони не змінювали свої вагові коефіцієнти
    for param in model.parameters():
        param.requires_grad = False

    # Замінюємо останній (класифікаційний) шар під нашу кількість класів (дій водія)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


# ==========================================
# 3. НАВЧАННЯ ШТУЧНОЇ НЕЙРОННОЇ МЕРЕЖІ
# ==========================================
def train_model(model, dataloader, epochs=5):
    print("--- Початок навчання ---")
    criterion = nn.CrossEntropyLoss()  # Функція втрат для класифікації
    # Оптимізуємо лише параметри останнього шару
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    model.train()  # Переводимо модель у режим навчання

    for epoch in range(epochs):
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()  # Обнуляємо градієнти

            # Прямий прохід
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Зворотне поширення помилки (Backpropagation)
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        print(f"Епоха {epoch + 1}/{epochs} | Втрати (Loss): {epoch_loss:.4f} | Точність (Accuracy): {epoch_acc:.4f}")

    print("Навчання завершено!")
    return model


# ==========================================
# 4. ЗАСТОСУВАННЯ НЕЙРОМЕРЕЖІ (Інференс)
# ==========================================
def predict_distraction(model, image_path, class_names):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не знайдено для тестування.")
        return

    model.eval()  # Переводимо модель у режим тестування (вимикає Dropout)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Додаємо розмірність батчу

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    print(f"\nРезультат аналізу фото водія:")
    print(f"Виявлено: {predicted_class.upper()} (Впевненість: {confidence.item() * 100:.2f}%)")


# ==========================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"ПОМИЛКА: Не знайдено папку '{DATASET_PATH}'.")
        print(
            "Створіть її, додайте всередину підпапки з класами (напр., 'safe', 'phone', 'drink') та покладіть туди фото.")
    else:
        # 1. Підготовка
        dataloader, class_names = prepare_data(DATASET_PATH)
        print(f"Знайдено класи: {class_names}")

        # 2. Конструювання
        model = build_model(len(class_names))

        # 3. Навчання
        trained_model = train_model(model, dataloader, epochs=3)

        # 4. Застосування (Візьмемо для перевірки перше-ліпше фото з нашого ж датасету)
        # Отримуємо шлях до першого файлу в першій підпапці
        test_folder = os.path.join(DATASET_PATH, class_names[0])
        test_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]

        if test_images:
            test_image_path = os.path.join(test_folder, test_images[0])
            predict_distraction(trained_model, test_image_path, class_names)
