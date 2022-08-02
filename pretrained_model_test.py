from torchvision.models import resnet50, ResNet50_Weights
from dataset_test import dataset
from torch.utils.data import DataLoader

weights = ResNet50_Weights.DEFAULT
model = resnet50(pretrained=True)
model.eval()

dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

to_test = 200
i = 0

while i in range(to_test):
    data = next(iter(dataloader))
    i += 1
pred = model(data[0])
class_id = pred.argmax().item()


print(weights.meta["categories"][class_id])