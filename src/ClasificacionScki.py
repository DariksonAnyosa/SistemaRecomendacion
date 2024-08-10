#Scki-learn es una libreria que nos permite hacer machine learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #Dividir los datos en conjunto de entrenaiento y prueba
from sklearn.preprocessing import OneHotEncoder, LabelEncoder#Preprocesamiento de datos, convertir datos en numericos
from sklearn.ensemble import RandomForestClassifier #se utiliza para crear y entrenar un modelo de clasificación basado en árboles de decisión
from sklearn.metrics import classification_report


datos = pd.read_csv('data/SistemaRecomendacion.csv')
df = pd.DataFrame(datos)#Convertir datos en un DataFrame
#print(df)

#Realizamos el preprocesamiento de datos
#Convertir los valores en numeros usando LabelEncoder
num = LabelEncoder()
df['genero_usuario'] = num.fit_transform(df['genero_usuario']) # Se codifica "H" como 0 y "M" como 1
df['metodo_pago'] = num.fit_transform(df['metodo_pago'])
df['frecuencia_compras'] = num.fit_transform(df['frecuencia_compras'])
df['dispositivo_uso'] = num.fit_transform(df['dispositivo_uso'])

# Convertir la variables de 'ubicacion_usuario' en variables dummy usando OneHotEncoder
ohe = OneHotEncoder()
ubicacion_encoder = ohe.fit_transform(df[['ubicacion_usuario']]).toarray() #Se convierte el resultado en un arraynumpy
ubicacion_df = pd.DataFrame(ubicacion_encoder, columns=ohe.get_feature_names_out(['ubicacion_usuario']), dtype='int') #get_feature_names_out especificamos que queremos los nombres de las caractericas para la columna ubicacion_usuario
df = df.join(ubicacion_df)  # Añadimos las columnas de ubicaciones codificadas al DataFrame original

# Seleccionar características relevantes para la clasificación
caracteristica = df[['edad_usuario', 'genero_usuario', 'precio_producto', 'opiniones_usuarios', 'descuentos_aplicados', 'metodo_pago', 'dispositivo_uso']]
caracteristica = caracteristica.join(ubicacion_df)  # Añadir las columnas de ubicaciones codificadas
labels = df['frecuencia_compras']  #Creamos la variable label para almacenar nuestra etiqueta relevante sera frecuencia compras


print(caracteristica)

# Dividir los datos en conjuntos de entrenamiento y prueba / el 20% de datos se utilizara como conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(caracteristica, labels, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
clf = RandomForestClassifier()
a=clf.fit(X_train, y_train)  # Ajustar el modelo con los datos de entrenamiento

# Hacer predicciones y evaluar el modelo
y_pred = clf.predict(X_test)  # Predecir las etiquetas para los datos de prueba
print(classification_report(y_test, y_pred))  # Generar un informe de clasificación que muestra la precisión, recall y F1-score

#f1=2*((recall*precision)/(recall+precision))
###########################################################################
