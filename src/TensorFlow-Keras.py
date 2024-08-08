import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('data/SistemaRecomendacion.csv')

# Usar CountVectorizer para la columna 'historial_compras'
vectorizer = CountVectorizer()
historial_compras_vectorizado = vectorizer.fit_transform(data['historial_compras'].fillna('')).toarray()

# Eliminar la columna original y añadir las nuevas columnas vectorizadas
data = data.drop(['historial_compras','nombre_producto', 'nombre_usuario'], axis=1)
data = pd.concat([data, pd.DataFrame(historial_compras_vectorizado)], axis=1)

# Convertir variables categóricas en números utilizando LabelEncoder
le = LabelEncoder()

# Incluir todas las columnas categóricas relevantes
categorical_columns = ['categoria_producto', 'genero_usuario', 'ubicacion_usuario', 'metodo_pago', 'dispositivo_uso','frecuencia_compras','fecha_venta']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Asegurarse de que todos los nombres de las columnas sean cadenas
data.columns = data.columns.astype(str)

# Separar características (X) y etiquetas (y)
X = data.drop('recomendado', axis=1)
y = data['recomendado']

# Escalar los datos para que estén en un rango similar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42)

# Construir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Usamos sigmoid porque es un problema binario
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# Mostrar un resumen del modelo
model.summary()

# Implementar EarlyStopping para detener el entrenamiento si no hay mejoras
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10000, batch_size=10000, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Guardar el modelo para uso futuro
model.save('modelo_recomendacion_mejorado.h5')

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}')

# Hacer predicciones
predicciones = model.predict(X_test)
predicciones_binarias = (predicciones > 0.5).astype(int)  # Convertir probabilidades en etiquetas binarias

# Ver algunas predicciones
print("Predicciones vs Valores Reales:")
for i in range(10):
    print(f'Predicción: {predicciones_binarias[i][0]}, Valor Real: {y_test.values[i]}')

# Gráfica del historial de entrenamiento
plt.figure(figsize=(14, 6))

# Gráfico de la pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de la precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Evolución de la precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.show()
