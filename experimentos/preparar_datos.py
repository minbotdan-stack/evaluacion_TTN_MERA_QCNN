
import numpy as np
from sklearn.datasets import load_iris

def cargar_datos():

    # Cargar el conjunto de datos Iris
    iris = load_iris()

    # Seleccionar versicolor y virginica
    X = iris.data[50:150]

    # Etiquetas originales:
    # 1 = versicolor
    # 2 = virginica
    y = iris.target[50:150]

    # Convertir etiquetas:
    # -1 = versicolor
    # +1 = virginica
    y = np.where(y == 1, -1, 1)

    return X, y


if __name__ == "__main__":

    X, y = cargar_datos()

    print("Dimensiones de X:", X.shape)
    print("Dimensiones de y:", y.shape)

    print("\nPrimeras cinco muestras:")
    print(X[:5])

    print("\nPrimeras cinco etiquetas:")
    print(y[:5])

    print("\nCantidad de muestras por clase:")
    print("Versicolor:", np.sum(y == -1))
    print("Virginica:", np.sum(y == 1))
