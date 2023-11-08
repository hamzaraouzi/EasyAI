from src.ModelsPreparers.imageClassificationModels.vit.swinTransformer.swinTransformer import (
    SwinTransformer,
)
from src.ModelsPreparers.imageClassificationModels.vit.swinTransformer.layers import (
    create_mask,
)
import torch

x = torch.randn((10, 3, 224, 224))
model = SwinTransformer(
    hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), model_name="ss"
)
y = model(x)
print(y.shape)
