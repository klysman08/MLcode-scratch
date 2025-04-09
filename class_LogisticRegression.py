# %%
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_proba = self._sigmoid(linear_model)
        y_predicted = [1 if p > 0.5 else 0 for p in y_predicted_proba]
        return np.array(y_predicted)

# Exemplo de uso:
if __name__ == '__main__':
    # Gerar dados de exemplo (binários)
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = np.where((X[:, 0] + X[:, 1]) > 1, 1, 0)

    # Criar e treinar o modelo de regressão logística
    model = LogisticRegression(learning_rate=1.1, n_iter=2000)
    model.fit(X, y)

    # Fazer previsões
    y_pred = model.predict(X)
    print("Previsões:", y_pred)

    # Avaliar a acurácia (simples)
    accuracy = np.mean(y_pred == y)
    print("Acurácia:", accuracy)

    # Prever para novos pontos
    new_points = np.array([[0.2, 0.3], [0.8, 0.9], [0.1, 0.8]])
    new_predictions = model.predict(new_points)
    print("Previsões para novos pontos:", new_predictions)
# %%
