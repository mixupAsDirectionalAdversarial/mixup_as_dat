'''Multi-layer Perceptron in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    # https://towardsdatascience.com/multi-layer-perceptron-usingfastai-and-pytorch-9e401dd288b8
    # https://stackoverflow.com/questions/48477198/problems-with-pytorch-mlp-when-training-the-mnist-dataset-retrieved-from-keras

    def __init__(self, for_dataset, num_classes=10, hidden_layer_dim=2000, dropout_ratio_=0): #, hidden_layer_dim=2000):
        super(MLPNet, self).__init__()

        self.for_dataset = for_dataset
        if self.for_dataset == "cifar10" or self.for_dataset == "cifar100":
            input_layer_dim = 3072
        elif self.for_dataset == "mnist":
            input_layer_dim = 784
        else:
            print("ERROR: unsupported dataset: ", args.dataset)
            exit(1)
        
        self.dropout_ratio = dropout_ratio_

        # https://blog.csdn.net/VictoriaW/article/details/73166752
        self.input_layer = nn.Linear(input_layer_dim, hidden_layer_dim, bias=True)
        nn.init.kaiming_normal_(self.input_layer.weight.data)
        self.num_layer = 1
        second_layer_dim = 200
        if self.num_layer == 2:
            self.hidden_layer = nn.Linear(hidden_layer_dim, second_layer_dim, bias=True)
            nn.init.kaiming_normal_(self.hidden_layer.weight.data)
            self.second_hidden_layer = nn.Linear(second_layer_dim, num_classes, bias=True)
            nn.init.kaiming_normal_(self.second_hidden_layer.weight.data)
        else:
            self.hidden_layer = nn.Linear(hidden_layer_dim, num_classes, bias=True)
            nn.init.kaiming_normal_(self.hidden_layer.weight.data)

        self.dropout_layer = nn.Dropout(self.dropout_ratio)

    def forward(self, input_data):
        out = input_data.view(input_data.size(0), -1)
        out = F.relu(self.input_layer(out))
        out = self.hidden_layer(out)
        if self.num_layer == 2:
            out = F.relu(out)
            out = self.second_hidden_layer(out)
        if self.dropout_ratio > 0:
            out = self.dropout_layer(out)

        return out
