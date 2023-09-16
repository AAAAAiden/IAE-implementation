import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, 100, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(100, 50, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(50, 25, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, 100, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(100, 50, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(50, 25, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, 100, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(100, 50, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(50, 25, kernel_size=1)
            # No activation for the last layer as mentioned
        )
    
    def forward(self, x):
        return self.layers(x)
