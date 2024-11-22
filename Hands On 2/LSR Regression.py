class SimpleLinearRegression:
    def __init__(self):
        # Dataset 
        self.data = {
            "Sales": [1,2,3,4,5,6,7,8,9],
            "Advertising": [2,4,6,8,10,12,14,16,18]
        }
        self.beta_0 = 0
        self.beta_1 = 0

    def calculate_betas(self):
        x = self.data["Advertising"]
        y = self.data["Sales"]
        n = len(x)

        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum([xi ** 2 for xi in x])
        sum_xy = sum([x[i] * y[i] for i in range(n)])

        # beta_1 y beta_0
        denominator = n * sum_x2 - sum_x ** 2
        self.beta_1 = (n * sum_xy - sum_x * sum_y) / denominator
        self.beta_0 = (sum_y * sum_x2 - sum_x * sum_xy) / denominator

    def predict(self, advertising):
        return self.beta_0 + self.beta_1 * advertising

    def print_regression_equation(self):
        # Imprime ecuacion
        print(f"La ecuación de regresión es: y = {self.beta_0:.2f} + {self.beta_1:.2f}x")

    def run(self):
        #betas
        self.calculate_betas()
        self.print_regression_equation()

        # Solicitar valores
        try:
            advertising = float(input("Ingrese el valor de Advertising para predecir las ventas: "))
            prediction = self.predict(advertising)
            print(f"Para Advertising = {advertising}, se predice Sales = {prediction:.2f}")
        except ValueError:
            print("Por favor, ingrese un valor numérico válido.")

# Clase principal
if __name__ == "__main__":
    model = SimpleLinearRegression()
    model.run()
