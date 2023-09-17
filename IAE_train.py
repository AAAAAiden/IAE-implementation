import torch
import torch.optim as optim
from IAE_model import Encoder, Decoder, Discriminator
import torch.nn as nn
import numpy as np
import sys
torch.autograd.set_detect_anomaly(True)

sys.stdout = open('training_log.txt', 'a')

# Data Loading
windows_ma = np.load('data/windows_ma.npy', allow_pickle=True).tolist()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
windows_ma_tensor = torch.FloatTensor(windows_ma).to(device)

# Model Initialization
m = 20  # or 100, as per the paper for synthetic or real data cases


# Parameters
lambda_coef = 10
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
mu = 0.1
num_epochs = 1000
nc = 5  
B = 64   
k = 3  # From n = 3m
sequence_length = k * m
encoder = Encoder(m).to(device)

decoder = Decoder(m).to(device)
discriminator = Discriminator().to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=alpha, betas=(beta1, beta2))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=alpha, betas=(beta1, beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=alpha, betas=(beta1, beta2))
mse_loss = nn.MSELoss()



encoder_buffer = []

for epoch in range(num_epochs):
    for _ in range(nc):
        for i in range(0, len(windows_ma_tensor) - B + 1, B):
            xi_batch = windows_ma_tensor[i:i+B]

            
            epsilon = torch.rand(1).item()

            v_hat = encoder(xi_batch)
            u = torch.rand_like(v_hat) * 2 - 1  
            # print('u shape' + str(u.size()))
            encoder_buffer.extend(v_hat.detach().cpu().tolist())
            if len(encoder_buffer) >= sequence_length:
                decoder_input = torch.FloatTensor(encoder_buffer[:sequence_length]).to(device)
                encoder_buffer = encoder_buffer[m:]


            v_bar = epsilon * u + (1 - epsilon) * v_hat

            out_vhat = discriminator(v_hat)
            # print('v_hat shape' + str(v_hat.size()))            
            out_u = discriminator(u)

            v_bar.requires_grad_(True)
            out_vbar = discriminator(v_bar)
            gradients = torch.autograd.grad(outputs=out_vbar, inputs=v_bar, grad_outputs=torch.ones(out_vbar.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_discriminator = out_vhat.mean() - out_u.mean() + lambda_coef * gradient_penalty

            discriminator_optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator_optimizer.step()

        # Encoder and Decoder update
        # xi_batch = windows_ma_tensor[:B]
        # encoded = encoder(xi_batch)
        # decoded = decoder(encoded)

        # loss_encoder = -discriminator(encoded).mean() + mu * mse_loss(decoded, xi_batch)
        # loss_decoder = mu * mse_loss(decoded, xi_batch)

        # encoder_optimizer.zero_grad()
        # loss_encoder.backward(retain_graph=True)
        # encoder_optimizer.step()

        # decoder_optimizer.zero_grad()
        # loss_decoder.backward()
        # decoder_optimizer.step()




        # Encoder update
        xi_batch = windows_ma_tensor[:B]
        encoded_for_encoder = encoder(xi_batch)
        decoded_for_encoder = decoder(encoded_for_encoder)  # Compute the decoded value here
        loss_encoder = -discriminator(encoded_for_encoder).mean() + mu * mse_loss(decoded_for_encoder, xi_batch)

        encoder_optimizer.zero_grad()
        loss_encoder.backward()
        encoder_optimizer.step()

        # Decoder update
        encoded_for_decoder = encoder(xi_batch)
        decoded = decoder(encoded_for_decoder)
        loss_decoder = mu * mse_loss(decoded, xi_batch)

        decoder_optimizer.zero_grad()
        loss_decoder.backward()
        decoder_optimizer.step()



    print(f"Epoch {epoch+1}/{num_epochs} - Loss Discriminator: {loss_discriminator.item()}, Loss Encoder: {loss_encoder.item()}, Loss Decoder: {loss_decoder.item()}")
