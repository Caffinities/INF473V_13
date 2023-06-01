import cutout
import torch
import torchvision
import torch.nn as nn

class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(2048, 1024),
                                                nn.ReLU(),
                                                nn.Linear(1024, num_classes),)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
model = ResNetFinetune(48)
optimizer = torch.optim.SGD(model.parameters(),lr=3e-2,momentum=0.9,weight_decay=0.001,nesterov=True)
lambda1 = lambda epoch: torch.cos(torch.tensor(7*3.1416*epoch/16/300))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

for i in range(30):
    print(scheduler.get_last_lr())
    optimizer.step()
    scheduler.step()
print(3e-2*torch.cos(torch.tensor(7*3.1416*1/16/300)))
    