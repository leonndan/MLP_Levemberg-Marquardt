import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from matplotlib.colors import ListedColormap


import matplotlib, numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression


from tkinter import *
from tkinter import ttk, Text
from tkinter.colorchooser import askcolor



# Lee los datos desde un archivo txt
def leer_datos():
    global ax,f,X,y
    datos = np.loadtxt(r'datos.txt')
    X = datos[:, :-1]  # Características
    y = datos[:, -1]   # Clases
    colores=["blue","red"]
    y_int = y.astype(int)
    cmap = ListedColormap(
            [colores[i] for i in np.unique(y_int)]
        )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    f.canvas.draw()


# Entrenar la red neuronal con el algoritmo Levenberg-Marquardt
def entrenar_red():
    #AQUI DIBUJAR EN EL CANVAS PARA QUE SE REFLEJEN, OPCIONAL QUE FUNCIONE POR CLICKS, RECUERDA PONER LOS MAX ITER COMO VARIABLE
    #LA CANTIDAD DE NEURONAS TAMBIEN PUEDE PONERSE COMO VARIABLES Y EL ALPHA ES EL LEARNING RATE
    global X,y,ax,f
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='lbfgs',alpha=0.3)
    clf.fit(X, y)
    return clf

def visualizar_inicio(x,y):
   global ax,f



# Visualizar los datos y la clasificación de la red neuronal
def visualizar_resultados(X, y, clf):
    global ax,f
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
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    f.canvas.draw()

# Nombre del archivo de datos





def inicio(root):
        global X,y,ax,f
        root.title('Red Neuronal Multicapa')
        # Crear cajas de texto para parámetros
        lr_label = tk.Label(root, text='Learning Rate:')
        lr_label.pack()
        lr_label.place(relx=0.75, rely=0.05, height=20, width=150)
        lr_entry = tk.Entry(root)
        lr_entry.insert(0, '0.3')
        lr_entry.pack()
        lr_entry.place(relx=0.75, rely=0.10, height=20, width=150)

        epoch_label = tk.Label(root, text='Épocas:')
        epoch_label.pack()
        epoch_label.place(relx=0.75, rely=0.15, height=20, width=100)
        epoch_entry = tk.Entry(root)
        epoch_entry.insert(0, '10000')
        epoch_entry.pack()
        epoch_entry.place(relx=0.75, rely=0.2, height=20, width=150)

        display_labels = ['0 - blue', '1 - red']

        train_button = tk.Button(root, text='Cargar Puntos', command=leer_datos)
        train_button.pack()
        train_button.place(relx=0.75, rely=0.40, height=30, width=150)

        train_button = tk.Button(root, text='Entrenar', command=entrenar_red)
        train_button.pack()
        train_button.place(relx=0.75, rely=0.30, height=30, width=150)

        

        f = Figure(figsize=(0,0), dpi=100)
        ax = f.add_subplot(111)
        ax.grid('on')
        ax.set_xlim([-4,4])
        ax.set_ylim([-4,4])
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.get_tk_widget().place(relx=0.02, rely=0.02, relheight=.7, relwidth=0.7)

def main():
    root = tk.Tk()
    app = inicio(root)
    root.mainloop()

if __name__ == "__main__":
    main()
