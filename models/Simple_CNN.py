import torch
import torch.nn as nn

class Simple_CNN(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AvgPool2d(16, stride=8),
            nn.Flatten(start_dim=1),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, input):
        return  self.cnn(input)

class Simple_CNN_tiles(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(7, stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, input):
        return  self.cnn(input)
