import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('data/SistemaRecomendacion.csv')


columns_to_keep = ['nombre_producto', 'categoria_producto', 'nombre_usuario', 'ubicacion_usuario', 'historial_compras','cantidad_vendida','opiniones_usuarios','visitas_producto','recomendado']
data = data[columns_to_keep]

# Convertir datos categóricos a numéricos
label_encoders = {}
categorical_columns = ['nombre_producto', 'categoria_producto', 'nombre_usuario', 'ubicacion_usuario', 'historial_compras', 'opiniones_usuarios','cantidad_vendida','visitas_producto']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Guardar el índice original antes de la división
data_index = data.index

# Separar características y etiquetas
X = data.drop('recomendado', axis=1)
y = data['recomendado']

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, data_index, test_size=0.2, random_state=42)

# Definir el modelo con más neuronas
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = modelo.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.5)

# Evaluar el modelo
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f'Pérdida en datos de prueba: {loss:.4f}')
print(f'Precisión en datos de prueba: {accuracy:.4f}')

# Graficar precisión y pérdida
plt.figure(figsize=(12, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()

# Realizar predicciones y mostrar recomendaciones
num_recommendations = 10  # Número de recomendaciones a mostrar
X_test_subset = X_test[:num_recommendations]
test_indices = test_index[:num_recommendations]
productos_recomendados = data.loc[test_indices].copy()
productos_recomendados['recomendacion'] = modelo.predict(X_test_subset)
productos_recomendados['recomendacion'] = productos_recomendados['recomendacion'].apply(lambda x: 'Recomendado' if x >= 0.5 else 'No recomendado')

# Mostrar las primeras recomendaciones
print(productos_recomendados.head(num_recommendations))
