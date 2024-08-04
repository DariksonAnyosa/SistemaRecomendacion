import pandas as pd
import numpy as np

df = pd.read_csv('data/SistemaRecomendacion.csv')
print(df.head())

tabla_x = df.drop(columns=['opiniones_usuarios', 'carrito_abandonado', 'descuentos_aplicados', 'metodo_pago','frecuencia_compras','dispositivo_uso'])
print(tabla_x)

columna_dummies = ['genero_usuario','categoria_producto','nombre_producto','ubicacion_usuario']

tabla_x = pd.get_dummies(tabla_x, columns=columna_dummies)
print(tabla_x)