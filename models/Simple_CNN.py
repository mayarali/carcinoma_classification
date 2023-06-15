import torch
import torch.nn as nn
from torchvision.transforms.functional import pad
class Simple_CNN(nn.Module):
    def __init__(self, encs=None, args=None):
        super().__init__()
        self.args = args
        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout

        self.cnn = encs[0]

        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_inner, num_classes)
        )

    def forward(self, input):
        return  self.fc(self.cnn(input))

class CNN_part(nn.Module):
    def __init__(self, encs=None, args=None):
        super().__init__()
        self.args = args

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AvgPool2d(32, stride=16),
            nn.Flatten(start_dim=1))

    def forward(self, input):
        return self.cnn(input)

class Simple_CNN_2Binary(nn.Module):
    def __init__(self, encs, args=None):
        super().__init__()
        self.args = args
        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes
        dropout = args.dropout

        self.cnn_cancer = encs[0]
        self.cnn = encs[1]

        # self.cnn = nn.Sequential(
        #     nn.BatchNorm2d(3),
        #     nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.AvgPool2d(32, stride=16),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, num_classes-1)
        # )

    def forward(self, input):
        # results = torch.zeros(input.shape[0], 3).cuda()

        out_full = self.cnn(input)
        out_full = pad(out_full, padding=(0, 0, 1, 0))

        if (out_full.argmax(dim=1) == 1).sum() > 0:
            out_cancer = self.cnn_cancer(input[out_full.argmax(dim=1) == 1])
            out_full[out_full.argmax(dim=1) == 1][:, 1:] = out_cancer

        return  out_full

