import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 48 -> 24

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 24 -> 12

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 12 -> 6

            # nn.Conv2d(256, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0), # 16 -> 8

            # nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0), # 8 -> 4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.ReLU(),

            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.7),
            # nn.ReLU(),

            nn.Linear(512, 7)
        )

    def forward(self, x):
        # CNN Layer
        x = self.cnn_layer(x)
        
        # Flatten
        x = x.flatten(1)

        # Fully Connected Layer
        x = self.fc_layers(x)

        return x