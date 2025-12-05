#!/usr/bin/env python3
"""Final test for Conv-Adapter integration."""

import sys
import os

# Убедимся, что импортируем наш модифицированный ultralytics
print("Testing direct integration in ultralytics...")

# Test 1: Import from ultralytics
try:
    from ultralytics.ultralytics.nn.modules import ConvAdapter, Add
    print("✅ Successfully imported ConvAdapter and Add from ultralytics")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you added the classes to:")
    print("  - ultralytics/nn/modules/conv.py")
    print("  - ultralytics/nn/modules/block.py")
    print("  - ultralytics/nn/modules/__init__.py")
    sys.exit(1)

# Test 2: Create instances
import torch

print("\nTesting ConvAdapter creation...")
adapter = ConvAdapter(c1=64, c2=128, k=3, s=1, gamma=4)
x = torch.randn(2, 64, 32, 32)
y = adapter(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {y.shape}")
print(f"  Parameters: {sum(p.numel() for p in adapter.parameters()):,}")
print("  ✅ ConvAdapter works")

print("\nTesting Add layer...")
add_layer = Add()
x1 = torch.randn(2, 128, 32, 32)
x2 = torch.randn(2, 128, 32, 32)
y = add_layer([x1, x2])
print(f"  Inputs: {x1.shape}, {x2.shape}")
print(f"  Output: {y.shape}")
print("  ✅ Add works")

# Test 3: YAML parsing
print("\nTesting YAML parsing...")
try:
    from ultralytics import YOLO
    
    # Load model with adapters
    model = YOLO('yolov8_adapter_test.yaml')
    print("  ✅ YAML parsed successfully")
    
    # Check for adapters
    adapters_found = []
    for name, module in model.model.named_modules():
        if isinstance(module, ConvAdapter):
            adapters_found.append(name)
    
    print(f"  Found {len(adapters_found)} ConvAdapter(s)")
    for adapter_name in adapters_found:
        print(f"    - {adapter_name}")
    
    if len(adapters_found) == 0:
        print("  ⚠️  No adapters found! Check YAML format")
    
except Exception as e:
    print(f"  ❌ YAML parsing failed: {e}")
    print("  Make sure parse_model() in ultralytics/nn/tasks.py is updated")

print("\n" + "="*50)
print("🎉 If all tests passed, integration is successful!")
print("\nNext steps:")
print("1. Create your full YAML config with adapters")
print("2. Train with: model = YOLO('your_config.yaml')")
print("3. Freeze backbone: for param in model.parameters(): param.requires_grad = False")
print("4. Unfreeze adapters: for name, param in model.named_parameters(): if 'adapter' in name: param.requires_grad = True")