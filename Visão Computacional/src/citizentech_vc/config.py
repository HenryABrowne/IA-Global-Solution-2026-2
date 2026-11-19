"""Configuração única do experimento (usada pelo notebook e por tools/train.py)."""

SEED = 42
IMG = 224
N_PER_CLASS = 6000  # amostra de 12 mil imagens (a base tem 40 mil)

# nome do modelo -> arquitetura + hiperparâmetros
CONFIGS = {
    "mobilenet_v3_large": dict(arch="mobilenet_v3_large", epochs=5, batch_size=64, lr=1e-3),
    "resnet18": dict(arch="resnet18", epochs=5, batch_size=64, lr=5e-4),
    "resnet50": dict(arch="resnet50", epochs=3, batch_size=32, lr=3e-4),
    # Melhoria pós-análise de robustez: mesma ResNet18 + aumento de condições de campo
    "resnet18_campo": dict(arch="resnet18", epochs=6, batch_size=64, lr=5e-4, field_aug=True),
}
