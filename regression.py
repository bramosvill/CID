import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LRS
class SimpleLinearRegression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.beta_0 = None
        self.beta_1 = None
    
    # parametros
    def fit(self):
        X_mean = np.mean(self.X)
        Y_mean = np.mean(self.Y)
        
        # beta_1
        numerator = np.sum((self.X - X_mean) * (self.Y - Y_mean))
        denominator = np.sum((self.X - X_mean) ** 2)
        self.beta_1 = numerator / denominator
        
        # beta_0
        self.beta_0 = Y_mean - self.beta_1 * X_mean
    
    # prediccion
    def predict(self, X):
        return self.beta_0 + self.beta_1 * X
    
    # MSE
    def mean_squared_error(self):
        predictions = self.predict(self.X)
        mse = np.mean((self.Y - predictions) ** 2)
        return mse

    # visualizacion recta
    def plot(self):
        plt.scatter(self.X, self.Y, color='blue', label='Datos reales')
        plt.plot(self.X, self.predict(self.X), color='red', label='Recta de regresión')
        plt.xlabel('Publicidad (Advertising)')
        plt.ylabel('Ventas (Sales)')
        plt.legend()
        plt.show()

# Datos

data = {
    'Sales': [2,4,6,8,10,12,14,16,18],
    'Advertising': [1,2,3,4,5,6,7,8,9]
}

df = pd.DataFrame(data)
X = df['Advertising']
Y = df['Sales']


model = SimpleLinearRegression(X, Y)
model.fit()

# Mostrar los resultados
print(f"Beta_0 (intercepto): {model.beta_0}")
print(f"Beta_1 (pendiente): {model.beta_1}")

# Calcular el error cuadrático medio (MSE)
mse = model.mean_squared_error()
print(f"Error cuadrático medio (MSE): {mse}")

# Visualizar la recta de regresión
model.plot()
