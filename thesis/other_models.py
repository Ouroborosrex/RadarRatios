import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes=19):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(1000, 128, kernel_size=5, stride=1, padding='same')
        # self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same')
        # self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=2, stride=1, padding='same')
        # self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Assuming binary classification

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        # print("After conv1: ", x.size())
        # x = self.pool1(x)
        # print("After pool1: ", x.size())
        x = nn.functional.relu(self.conv2(x))
        # print("After conv2: ", x.size())
        # x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        # x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dense = nn.Linear(498, 31872)
        self.fc = nn.Linear(31872, 8)  # The output size here depends on your specific task

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dense(x)
        x = self.fc(x)
        return x

# create an instance of the Net
# net = Net()
import torch
from torch import nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class WaveformClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(WaveformClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(1, d_model)
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


