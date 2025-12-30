import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import ModeloCNN
# from models.mlp import ModeloMLP

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
# Dataset de teste
# --------------------------------------------------
testset = datasets.MNIST(
    root="./data/MNIST_data",
    train=False,
    download=True,
    transform=transform
)

testloader = DataLoader(testset, batch_size=64, shuffle=False)

# --------------------------------------------------
# Função de avaliação
# --------------------------------------------------
def avaliar(modelo):
    modelo.to(device)
    modelo.eval()

    corretas = 0
    total = 0

    with torch.no_grad():
        for imagens, etiquetas in testloader:
            imagens = imagens.to(device)
            etiquetas = etiquetas.to(device)

            output = modelo(imagens)
            preds = output.argmax(dim=1)

            corretas += (preds == etiquetas).sum().item()
            total += etiquetas.size(0)

    print(f"Acurácia: {100 * corretas / total:.2f}%")

# --------------------------------------------------
# Execução
# --------------------------------------------------
if __name__ == "__main__":
    modelo = ModeloCNN()
    # modelo = ModeloMLP()  # alternativa

    modelo.load_state_dict(
        torch.load("checkpoints/modelo_cnn.pth", map_location=device)
    )

    avaliar(modelo)
