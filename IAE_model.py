import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, m):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(m, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 1)  # to get scalar output νt
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, n):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, n)  # to get scalar output xˆt
        )
    
    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 100),  # Assuming input is νt, a scalar
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25)  # Linear activation as the last one
        )
    
    def forward(self, x):
        return self.layers(x)
