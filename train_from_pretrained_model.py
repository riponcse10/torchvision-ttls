from torchvision.models import resnet50, ResNet50_Weights
from dataset_test import dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

weights = ResNet50_Weights.DEFAULT
model = resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features=in_features, out_features=102)
model.train().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
epochs = 20
batch = next(iter(dataloader))

for epoch in range(epochs):
    # epoch_loss = 0.0
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    epoch_loss = loss.item()
    print(epoch_loss)
    # for data in iter(dataloader):
