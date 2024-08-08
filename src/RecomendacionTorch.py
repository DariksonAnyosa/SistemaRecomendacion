import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Cargar datos
data_base = pd.read_csv('data/SistemaRecomendacion.csv')

# Eliminar columnas innecesarias
columnas_innecesarias = ['nombre_producto', 'categoria_producto', 'nombre_usuario', 'ubicacion_usuario', 'historial_compras', 'opiniones_usuarios', 'recomendado']
data = data_base[columnas_innecesarias]

# Convertir datos categóricos a numéricos
label_encoders = {}
categorical_columns = ['nombre_producto', 'categoria_producto', 'nombre_usuario', 'ubicacion_usuario', 'historial_compras', 'opiniones_usuarios']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separar características y etiquetas
X = data.drop('recomendado', axis=1)
y = data['recomendado']

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separo los datos de entrenamiento con los datos de prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir los datos a tensores de PyTorch
t_x_train = torch.from_numpy(X_train).float()
t_x_test = torch.from_numpy(X_test).float()
t_y_train = torch.from_numpy(y_train.values).float().unsqueeze(1)
t_y_test = torch.from_numpy(y_test.values).float().unsqueeze(1)

# Definir la red neuronal
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 1)

    def forward(self, inputs):
        pred_1 = torch.relu(self.linear1(inputs))
        pred_2 = torch.relu(self.linear2(pred_1))
        pred_3 = torch.relu(self.linear3(pred_2))
        pred_f = torch.sigmoid(self.linear4(pred_3))
        return pred_f

# Inicializar la red, la función de pérdida y el optimizador
n_entradas = t_x_train.shape[1]
model = Red(n_entradas)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(t_x_train)
    loss = criterion(y_pred_train, t_y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Gráficos de rendimiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Pérdida de entrenamiento')
plt.plot(range(5, epochs+1, 5), test_losses, label='Pérdida en prueba')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida a lo largo del entrenamiento')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(5, epochs+1, 5), accuracies, label='Precisión en prueba')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión a lo largo del entrenamiento')
plt.legend()

plt.tight_layout()
plt.show()

# Realizar predicciones y mostrar recomendaciones
num_recommendations = 10
predicciones_proba = y_pred_test[:num_recommendations].numpy().flatten() # Obtener las probabilidades
predicciones = y_pred_test[:num_recommendations].round().numpy().flatten() # Obtener las predicciones

# Crear un DataFrame con las recomendaciones usando los índices originales
indices_originales = y_test.index[:num_recommendations]
productos_recomendados = data_base.iloc[indices_originales].copy()
productos_recomendados['probabilidad_recomendacion'] = predicciones_proba
productos_recomendados['recomendacion'] = predicciones
productos_recomendados['recomendacion'] = productos_recomendados['recomendacion'].apply(lambda x: 'Recomendado' if x == 1 else 'No recomendado')

# Mostrar las primeras 10 recomendaciones incluyendo la ubicación del producto y la probabilidad
print("Productos recomendados:")
print(productos_recomendados[['nombre_producto', 'ubicacion_usuario', 'probabilidad_recomendacion', 'recomendacion']])
