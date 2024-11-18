import torch
import torchvision

model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)

print(predictions[0])
