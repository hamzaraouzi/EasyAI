from src.ModelsPreparers.imageClassificationModels.vit.swinTransformer.swinTransformer import (
    SwinTransformer,
)
import torch

x = torch.randn((1, 3, 256, 256))
model = SwinTransformer(
    hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), model_name="ss"
)
y = model(x)
print(y.shape)
