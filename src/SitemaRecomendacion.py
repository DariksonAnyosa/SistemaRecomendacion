# Importamos las librerias pandas y numpy
import pandas as pd
import numpy as np

# ENTREGABLE 01

# Aplicamos la funcion para leer csv y lo guardamos en la variable df
tabla_base = pd.read_csv('data/SistemaRecomendacion.csv')

# Imprimir cantidad de filas y el rango
indices = tabla_base.index
print("La cantidad de indices son: ",indices)

# Imprimir las etiquetas de las columnas
etiquetas = tabla_base.columns
print("Las etiquetas de la tabla son: ",etiquetas)

# Imprimir los tipos de datos de las columnas
tipos = tabla_base.dtypes
print("Los tipos de datos de la tabla son: ",tipos)

# Imprimir la cantidad de datos
datos = tabla_base.shape
print("La cantidad de dato y las columnas: ",datos)

###########################################################################
