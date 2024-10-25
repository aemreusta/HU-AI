import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        x = self.fc1(x)
        x = self.relu7(x)
        x = self.fc2(x)
        return x


class CustomDropCNN(CustomCNN):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomDropCNN, self).__init__(num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.fc2(x)
        return x


class EfficientNetB0Custom(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0Custom, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=True)
        in_features = self.efficientnet_b0.classifier[1].in_features
        self.efficientnet_b0.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.efficientnet_b0(x)
