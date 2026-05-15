import numpy as np

# 1. Підготовка даних
# 6 бінарних знаків (растр 3x3, витягнутий у вектор 1x9)
X = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Горизонтальна лінія (-)
    [0, 1, 0, 0, 1, 0, 0, 1, 0],  # Вертикальна лінія (|)
    [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Діагональ (\)
    [0, 0, 1, 0, 1, 0, 1, 0, 0],  # Діагональ (/)
    [0, 1, 0, 1, 1, 1, 0, 1, 0],  # Хрест (+)
    [1, 0, 1, 0, 1, 0, 1, 0, 1]  # Знак (X)
])

# Очікувані виходи (One-Hot Encoding: 1 для правильного класу, 0 для інших)
Y = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])


# 2. Конструювання нейромережі
class RawNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        # Ініціалізація ваг та зсувів
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    # 3. Прямий прохід (Застосування)
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    # 4. Навчання (Backpropagation)
    def train(self, X, Y, epochs=5000, learning_rate=0.5):
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)

            # Помилка
            error = Y - output

            if epoch % 1000 == 0:
                mse = np.mean(np.square(error))
                print(f"Епоха {epoch}: Помилка (MSE) = {mse:.5f}")

            # Backward (Градієнти)
            d_output = error * self.sigmoid_derivative(output)

            error_hidden = np.dot(d_output, self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.A1)

            # Оновлення ваг
            self.W2 += np.dot(self.A1.T, d_output) * learning_rate
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.W1 += np.dot(X.T, d_hidden) * learning_rate
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


# Створення та навчання
nn = RawNeuralNetwork(input_size=9, hidden_size=12, output_size=6)
print("--- Початок навчання ---")
nn.train(X, Y, epochs=5000, learning_rate=0.5)

# Доведення працездатності
print("\n--- Тестування ---")
predictions = nn.forward(X)
sign_names = ["(-)", "(|)", "(\\)", "(/)", "(+)", "(X)"]

for i in range(len(X)):
    pred_class = np.argmax(predictions[i])
    true_class = np.argmax(Y[i])
    confidence = predictions[i][pred_class] * 100
    print(f"Знак: {sign_names[true_class]} -> Передбачено: {sign_names[pred_class]} (Впевненість: {confidence:.2f}%)")
