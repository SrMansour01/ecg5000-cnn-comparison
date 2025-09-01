
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import openml
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import time
import seaborn as sns

# Parâmetros
epochs = 50
batch_size = 128
learning_rate = 0.001

# Carregar ECG5000
print("Baixando dataset ECG5000 do OpenML...")
dataset = openml.datasets.get_dataset(44794)
X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# Converter para numpy
X = X.to_numpy()
y = y.to_numpy().astype(int)
y -= y.min()  # Ajustar rótulos para começar em 0
num_classes = len(np.unique(y))
print(f"Classes: {np.unique(y)}")
print(f"Distribuição das classes: {np.bincount(y)}")

# Normalização por amostra
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tensores
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [N, 1, 140]
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# Calcular pesos para classes desbalanceadas
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)


# Visualizar espectrograma
def visualize_spectrogram(x, filename="spectrogram.png"):
    x = x.squeeze(1)
    seq_len = x.shape[1]
    n_fft, hop_length = 16, 4
    if n_fft > seq_len:
        n_fft = seq_len // 2
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                      return_complex=True, center=True, pad_mode='reflect',
                      window=window)
    spec = torch.abs(spec)
    plt.figure(figsize=(8, 4))
    plt.imshow(spec[0].numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Espectrograma de uma amostra")
    plt.xlabel("Tempo")
    plt.ylabel("Frequência")
    plt.savefig(filename)
    plt.close()


# CNN 1D
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 35, 100)  # 140 / 2 / 2 = 35
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# CNN 2D
def compute_spectrogram(x, n_fft=16, hop_length=4):
    x = x.squeeze(1)
    seq_len = x.shape[1]
    if n_fft > seq_len:
        n_fft = seq_len // 2
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                      return_complex=True, center=True, pad_mode='reflect',
                      window=window)
    spec = torch.abs(spec)
    spec = spec.unsqueeze(1)
    return spec


class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = compute_spectrogram(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Função de treinamento com medição de tempo
def train_model(model, train_loader, test_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_losses, test_accs = [], []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Avaliação
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        test_accs.append(correct / total)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_losses[-1]:.4f} - Test Acc: {test_accs[-1]:.4f}")
    train_time = time.time() - start_time

    # Medir tempo de inferência
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    inference_time = time.time() - start_time

    return train_losses, test_accs, train_time, inference_time


# Função para avaliar métricas detalhadas
def evaluate_model(model, test_loader, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Classification report
    print(f"\nClassification Report para {model_name}:")
    print(classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(num_classes)],
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.savefig(f"confusion_matrix_{model_name.lower()}.png")
    plt.close()


# Visualizar espectrograma de uma amostra
visualize_spectrogram(X_train[0:1], "spectrogram.png")

# Treinar CNN1D
print("Treinando CNN1D...")
model1d = CNN1D(num_classes)
loss1d, acc1d, train_time1d, inference_time1d = train_model(model1d, train_loader, test_loader, epochs, learning_rate)
evaluate_model(model1d, test_loader, "CNN1D")
print(f"Tempo de treino (CNN1D): {train_time1d:.2f} segundos")
print(f"Tempo de inferência (CNN1D): {inference_time1d:.2f} segundos")

# Treinar CNN2D
print("Treinando CNN2D...")
model2d = CNN2D(num_classes)
loss2d, acc2d, train_time2d, inference_time2d = train_model(model2d, train_loader, test_loader, epochs, learning_rate)
evaluate_model(model2d, test_loader, "CNN2D")
print(f"Tempo de treino (CNN2D): {train_time2d:.2f} segundos")
print(f"Tempo de inferência (CNN2D): {inference_time2d:.2f} segundos")

# Gráficos comparativos
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss1d, label="CNN1D")
plt.plot(loss2d, label="CNN2D")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc1d, label="CNN1D")
plt.plot(acc2d, label="CNN2D")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.legend()
plt.savefig("training_plots.png")
plt.close()