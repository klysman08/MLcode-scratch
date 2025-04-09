# %%

import numpy as np

class LinearRegression:
    def __init__(self):
        self.b0 = None
        self.b1 = None

    def estimate_coef(self, x, y):
        """
        Estima os coeficientes da regressão linear.

        Args:
            x (np.array): Variável independente.
            y (np.array): Variável dependente.

        Returns:
            tuple: Coeficientes b0 e b1.
        """
        n = np.size(x)
        m_x, m_y = np.mean(x), np.mean(y)
        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x
        self.b1 = SS_xy / SS_xx
        self.b0 = m_y - self.b1 * m_x
        return (self.b0, self.b1)

    def R2(self, x, y):
        """
        Calcula o coeficiente de determinação R².

        Args:
            x (np.array): Variável independente.
            y (np.array): Variável dependente.

        Returns:
            float: Valor de R².
        """
        n = len(y)
        c1 = 0
        c2 = 0
        ym = np.mean(y)
        for i in range(0, n):
            y_pred = self.b0 + x[i] * self.b1
            c1 = c1 + (y[i] - y_pred) ** 2
            c2 = c2 + (y[i] - ym) ** 2
        R2 = 1 - c1 / c2
        return R2


# Exemplo de uso
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
model = LinearRegression()
b0, b1 = model.estimate_coef(x, y)
print("Coeficientes:\nb_0 = {}  \nb_1 = {}".format(b0, b1))
print('R2:', model.R2(x, y))

# %%
