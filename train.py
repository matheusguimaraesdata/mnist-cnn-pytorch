import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time

from models.cnn import ModeloCNN
# from models.mlp import ModeloMLP  # use se quiser testar MLP

# --------------------------------------------------
# Configuração de dispositivo
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Transformações
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --------------------------------------------------
# Dataset e DataLoader
# --------------------------------------------------
trainset = datasets.MNIST(
    root="./data/MNIST_data",
    train=True,
    download=True,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# --------------------------------------------------
# Função de treino
# --------------------------------------------------
def treino(modelo, epochs=10):
    modelo.to(device)

    criterio = nn.NLLLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(
        otimizador, step_size=5, gamma=0.5
    )

    modelo.train()
    inicio = time()

    for epoch in range(epochs):
        perda_total = 0

        for imagens, etiquetas in trainloader:
            imagens = imagens.to(device)
            etiquetas = etiquetas.to(device)

            otimizador.zero_grad()
            output = modelo(imagens)
            loss = criterio(output, etiquetas)
            loss.backward()
            otimizador.step()

            perda_total += loss.item()

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {perda_total / len(trainloader):.4f}"
        )

    print(f"Tempo total: {(time() - inicio) / 60:.2f} minutos")

    # --------------------------------------------------
    # Salvando o modelo treinado
    # --------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(modelo.state_dict(), "checkpoints/modelo_cnn.pth")
    print("Modelo salvo em checkpoints/modelo_cnn.pth")

# --------------------------------------------------
# Execução
# --------------------------------------------------
if __name__ == "__main__":
    modelo = ModeloCNN()
    # modelo = ModeloMLP()  # alternativa
    treino(modelo, epochs=10)
