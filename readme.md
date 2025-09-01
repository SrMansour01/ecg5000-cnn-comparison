# ECG Classification with 1D and 2D CNNs
Este projeto implementa e compara duas arquiteturas de redes neurais convolucionais (1D e 2D) para classificação de sinais de eletrocardiograma (ECG) usando o dataset ECG5000 do OpenML.

## 📋 Descrição
O código realiza a classificação de sinais ECG em 5 categorias diferentes, comparando o desempenho de:

CNN 1D: Opera diretamente no sinal temporal bruto

CNN 2D: Opera em representações de espectrograma (transformada de Fourier de curta duração - STFT) do sinal

## 🚀 Funcionalidades
Pré-processamento e normalização de sinais ECG

Geração de espectrogramas via STFT

Implementação de arquiteturas CNN 1D e 2D

Treinamento com balanceamento de classes

Avaliação detalhada com métricas e matrizes de confusão

Comparação de desempenho entre abordagens

Visualização de resultados e espectrogramas

## 📊 Dataset
O projeto utiliza o ECG5000 dataset from OpenML, que contém:

5.000 sinais de ECG

140 pontos por amostra

5 classes de batimentos cardíacos

## 🛠️ Tecnologias Utilizadas
Python 3.x

PyTorch (para redes neurais)

NumPy (para processamento numérico)

Matplotlib/Seaborn (para visualizações)

Scikit-learn (para pré-processamento e métricas)

OpenML (para download do dataset)

## 📦 Instalação
Clone o repositório:
```bash
git clone https://github.com/seu-usuario/ecg-classification-cnn.git
cd ecg-classification-cnn
```
## 🎯 Como Executar
Execute o script principal:

```bash
python main.py
```

## 📈 Resultados
O projeto gera automaticamente:

Gráficos de loss e acurácia durante o treinamento

Matrizes de confusão para ambos os modelos

Espectograma do dataset

Relatórios de classificação detalhados

Comparação de tempo de treinamento e inferência



