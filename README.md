# MNIST – Classificação de Dígitos com MLP e CNN (PyTorch)

Projeto de Deep Learning para classificação de dígitos manuscritos utilizando o dataset **MNIST**, implementado em **PyTorch**.  
O projeto compara duas abordagens de redes neurais: **MLP (Multi-Layer Perceptron)** e **CNN (Convolutional Neural Network)**.

---

## 📌 Objetivo do Projeto

Demonstrar, de forma prática, a diferença de desempenho entre:

- **MLP**: rede neural totalmente conectada
- **CNN**: rede neural convolucional, mais adequada para imagens

Ambos os modelos são treinados e avaliados no mesmo dataset, permitindo comparação direta dos resultados.

---

## 📊 Resultados

- Dataset: MNIST (60.000 imagens de treino / 10.000 de teste)
- Melhor modelo: **CNN**
- Acurácia da CNN: **99.21%**
- Framework: PyTorch

---

## 🧠 Modelos Implementados

### 🔹 MLP (Multi-Layer Perceptron)

- Utiliza apenas camadas totalmente conectadas
- Requer achatamento da imagem (28x28 → 784)
- Serve como modelo base para comparação

### 🔹 CNN (Convolutional Neural Network)

- Utiliza camadas convolucionais e pooling
- Explora padrões espaciais das imagens
- Apresenta desempenho superior para tarefas de visão computacional

---

## 📁 Estrutura do Projeto
mnist-cnn-pytorch/
│
├── README.md              # Documentação do projeto
├── requirements.txt       # Dependências do projeto
├── .gitignore             # Arquivos e pastas ignorados pelo Git
├── train.py               # Script de treinamento do modelo
├── evaluate.py            # Script de avaliação do modelo treinado
│
├── models/                # Definições das arquiteturas de rede neural
│   ├── cnn.py             # Modelo Convolutional Neural Network (CNN)
│   └── mlp.py             # Modelo Multi-Layer Perceptron (MLP)

---

## ▶️ Como Executar o Projeto

### 1️⃣ Criar ambiente virtual (opcional, recomendado)

```python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 2️⃣ Instalar dependências

```pip install -r requirements.txt
```

### 3️⃣ Treinar o modelo

```python train.py```

### 4️⃣ Avaliar o modelo treinado

```python evaluate.py```

---

## 🛠 Tecnologias Utilizadas

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## 📌 Observações

- O MNIST é baixado automaticamente via torchvision
- Os pesos do modelo são salvos localmente após o treino
- O foco do projeto é aprendizado e comparação de arquiteturas

---

## 👤 Autor

**Matheus Guimarães**
Estudante de Análise e Desenvolvimento de Sistemas | Dados | IA
