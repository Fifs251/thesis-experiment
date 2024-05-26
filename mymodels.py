import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        counter=1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 
                      mean=0, std=0.01)
                if counter in [1,3]:
                    m.bias.data.fill_(0)
                else:
                    m.bias.data.fill_(0)
                counter+=1
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 
                      mean=0, std=0.01)
                m.bias.data.fill_(0)

class AlexNet_anti_DRY(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )
        torch.nn.init.normal_(self.features[0].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.features[3].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.features[6].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.features[8].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.features[10].weight, 
                      mean=0, std=0.01)


        torch.nn.init.normal_(self.classifier[1].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.classifier[4].weight, 
                      mean=0, std=0.01)
        torch.nn.init.normal_(self.classifier[6].weight, 
                      mean=0, std=0.01)

        self.features[0].bias.data.fill_(0)
        self.features[3].bias.data.fill_(1)
        self.features[6].bias.data.fill_(0)
        self.features[8].bias.data.fill_(1)
        self.features[10].bias.data.fill_(1)

        self.classifier[1].bias.data.fill_(1)
        self.classifier[4].bias.data.fill_(1)
        self.classifier[6].bias.data.fill_(1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_liu(nn.Module):

  def __init__(self, classes=100):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.Sigmoid(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.Sigmoid(),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.Sigmoid(),
      nn.Linear(4096, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


class AlexNet_no_weight_init(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_Tanh(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 1024),
                nn.Tanh(),
                nn.Linear(1024, num_classes),
            )
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        counter=1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 
                      mean=0, std=0.01)
                if counter in [1,3]:
                    m.bias.data.fill_(0)
                else:
                    m.bias.data.fill_(0)
                counter+=1
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 
                      mean=0, std=0.01)
                m.bias.data.fill_(0)

class AlexNet_Sigm(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 1024),
                nn.Tanh(),
                nn.Linear(1024, num_classes),
            )
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        counter=1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 
                      mean=0, std=0.01)
                if counter in [1,3]:
                    m.bias.data.fill_(0)
                else:
                    m.bias.data.fill_(0)
                counter+=1
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 
                      mean=2, std=0.01)
                m.bias.data.fill_(0)
