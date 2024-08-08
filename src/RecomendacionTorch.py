import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Se carga la base con la libreria pandas
data_base = pd.read_csv('data/SistemaRecomendacion.csv')

# Eliminar columnas innecesarias 
columnas_innecesarias = ['nombre_producto', 'categoria_producto', 'nombre_usuario', 'opiniones_usuarios','id_producto','id_usuario','genero_usuario','ubicacion_usuario','historial_compras','fecha_venta','descuentos_aplicados','metodo_pago','frecuencia_compras','dispositivo_uso']
data = data_base.drop(columnas_innecesarias, axis=1)

# Convertir datos categóricos a numéricos
le = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':  # Convertir solo columnas categóricas (de tipo objeto)
        data[i] = le.fit_transform(data[i])

# Separar características y etiquetas
X = data.drop('recomendado', axis=1)
y = data['recomendado']

# Escalar características con StandardScaler de sklearn
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separo los datos de entrenamiento con los datos de prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir los datos a tensores de PyTorch
t_x_train = torch.from_numpy(X_train).float()
t_x_test = torch.from_numpy(X_test).float()
t_y_train = torch.from_numpy(y_train.values).float().unsqueeze(1)
t_y_test = torch.from_numpy(y_test.values).float().unsqueeze(1)

# Definir la red neuronal con más capas y neuronas
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 1)

    def forward(self, inputs):
        pred_1 = torch.relu(self.linear1(inputs))
        pred_2 = torch.relu(self.linear2(pred_1))
        pred_3 = torch.relu(self.linear3(pred_2))
        pred_4 = torch.relu(self.linear4(pred_3))
        pred_5 = torch.relu(self.linear5(pred_4))
        pred_f = torch.sigmoid(self.linear6(pred_5))
        return pred_f

# Inicializar la red, la función de pérdida y el optimizador
n_entradas = t_x_train.shape[1]
model = Red(n_entradas)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Entrenar el modelo con más épocas
epochs = 1000
train_losses = []
test_losses = []
accuracies = []

print('Entrenando el modelo...')
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(t_x_train)
    loss = criterion(y_pred_train, t_y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_test = model(t_x_test)
            test_loss = criterion(y_pred_test, t_y_test)
            y_pred_test_class = y_pred_test.round()
            accuracy = (y_pred_test_class.eq(t_y_test).sum() / float(t_y_test.shape[0])).item()
            test_losses.append(test_loss.item())
            accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

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
