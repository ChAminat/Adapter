import sys
sys.path.insert(0, 'ultralytics')

from ultralytics.ultralytics.nn.modules.conv import ConvAdapter
import torch

# Простейшая проверка
adapter = ConvAdapter(64, 128)
x = torch.randn(1, 64, 32, 32)
y = adapter(x)
print(f"✅ ConvAdapter работает! Вход: {x.shape}, Выход: {y.shape}")
print(f"   Параметров: {sum(p.numel() for p in adapter.parameters()):,}")