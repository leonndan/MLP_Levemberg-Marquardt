import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from matplotlib.colors import ListedColormap

# Lee los datos desde un archivo txt
def leer_datos(nombre_archivo):
    datos = np.loadtxt(nombre_archivo)
    X = datos[:, :-1]  # Características
    y = datos[:, -1]   # Clases
    return X, y

# Entrenar la red neuronal con el algoritmo Levenberg-Marquardt
def entrenar_red(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, solver='lbfgs')
    clf.fit(X, y)
    return clf

# Visualizar los datos y la clasificación de la red neuronal
def visualizar_resultados(X, y, clf):
    
    plt.title('Entrenamiento Levemberg Marquardt')
    

    # Crear un plano cartesiano para visualizar la clasificación
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colores=["blue","red"]
    y_int = y.astype(int)
    cmap = ListedColormap(
            [colores[i] for i in np.unique(y_int)]
        )
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    plt.show()

# Nombre del archivo de datos
nombre_archivo = 'datos.txt'

# Leer datos
X, y = leer_datos(nombre_archivo)

# Entrenar la red neuronal
clf = entrenar_red(X, y)

# Visualizar resultados
visualizar_resultados(X, y, clf)