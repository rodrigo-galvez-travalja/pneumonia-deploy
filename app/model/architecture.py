"""
Arquitectura de la CNN para clasificación de radiografías de tórax.

Clasifica imágenes Chest X-Ray en 3 clases:
    - NORMAL
    - PNEUMONIA_BACTERIAL
    - PNEUMONIA_VIRAL
"""

import torch
import torch.nn as nn


class PneumoniaCNN(nn.Module):
    """
    Red neuronal convolucional para diagnóstico de neumonía.

    Arquitectura:
        - Conv2d(3→32) + ReLU + MaxPool2d(2,2)
        - Conv2d(32→64) + ReLU + MaxPool2d(2,2)
        - Conv2d(64→128) + ReLU + MaxPool2d(2,2)
        - Dropout(0.5)
        - Linear(128*12*12 → 128) + ReLU
        - Linear(128 → num_classes)

    Args:
        num_classes (int): Número de clases de salida. Por defecto 3.
    """

    def __init__(self, num_classes: int = 3):
        super(PneumoniaCNN, self).__init__()

        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Bloque convolucional 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling compartido entre los 3 bloques
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Regularización
        self.dropout = nn.Dropout(p=0.5)

        # Capas fully connected
        # Tras 3 MaxPool(2,2) sobre input 100x100: 100 → 50 → 25 → 12
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.

        Args:
            x: Tensor de entrada con shape (batch, 3, 100, 100)

        Returns:
            Tensor de logits con shape (batch, num_classes)
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = self.dropout(x)

        # Aplanar para las capas FC
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Nombres de las clases en el orden del modelo
CLASS_NAMES = ["NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRAL"]

# Etiquetas legibles para mostrar en la API y el frontend
CLASS_LABELS = {
    "NORMAL": "Normal",
    "PNEUMONIA_BACTERIAL": "Neumonía Bacteriana",
    "PNEUMONIA_VIRAL": "Neumonía Vírica"
}
