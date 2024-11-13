import numpy as np

# LSR
class SimpleLinearRegression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.beta_0 = 0
        self.beta_1 = 0
    
    def fit(self):
        # X y y
        X_mean = np.mean(self.X)
        Y_mean = np.mean(self.Y)
        
        #  beta_1
        numerator = np.sum((self.X - X_mean) * (self.Y - Y_mean))
        denominator = np.sum((self.X - X_mean) ** 2)
        self.beta_1 = numerator / denominator
        
        # beta_0
        self.beta_0 = Y_mean - self.beta_1 * X_mean
    
    def predict(self, X_new):
        return self.beta_0 + self.beta_1 * X_new
    
    def print_equation(self):
        # Imprimir la ecuación de la recta de regresión
        print(f"Ecuación de regresión: Y = {self.beta_0:.2f} + {self.beta_1:.2f} * X")
        
    def get_parameters(self):
        return self.beta_0, self.beta_1


# main
def main():

    X = [1,2,3,4,5,6,7,8,9]
    Y = [2,4,6,8,10,12,14,16,18]
    
    model = SimpleLinearRegression(X, Y)
    model.fit()
    
    #ecuacion
    model.print_equation()
    
    while True:
        try:
            X_new = float(input("Ingrese un valor de inversión en publicidad (X) para predecir las ventas (Y): "))
            Y_pred = model.predict(X_new)
            print(f"Predicción de ventas (Y) para X = {X_new}: {Y_pred:.2f}")
        except ValueError:
            print("Por favor ingrese un número válido para la inversión en publicidad.")
        except KeyboardInterrupt:
            print("\nTerminando el programa.")
            break


# Ejecutar la función principal
if __name__ == "__main__":
    main()
