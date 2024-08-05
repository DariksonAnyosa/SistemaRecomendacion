# Importamos las librerias pandas y numpy
import pandas as pd
import numpy as np

# ENTREGABLE 01

# Aplicamos la funcion para leer csv y lo guardamos en la variable df
df = pd.read_csv('data/SistemaRecomendacion.csv')
print(df.head())

# Luego hacemos un drop(eliminar) las siguientes columnas 
# Luego lo almacenamos en una nueva variable tabla_x
tabla_x = df.drop(columns=['carrito_abandonado', 'descuentos_aplicados', 'metodo_pago','frecuencia_compras','dispositivo_uso'])
print(tabla_x)

# Creo una variable y ingreso las siguientes columnas 
columna_dummies = ['genero_usuario','categoria_producto','nombre_producto','ubicacion_usuario']

# Luego aplico la funcion dummies a la tabla_x a las columnas de la variable columna_dummies
# Funcion dummies dependiendo de cantidad de datos crea nuevas columnas 
tabla_x = pd.get_dummies(tabla_x, columns=columna_dummies)
print(tabla_x)

###########################################################################

# ENTREGABLE 02

# Scki-learn es una libreria que nos permite hacer machine learning




# Pytorch es una libreria que nos permite hacer deep learning
