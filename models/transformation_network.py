import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder for the Autoencoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# Define the Decoder for the Autoencoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Assuming output is in [0, 1]
        return x

# Define the Autoencoder
class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        #check if x shape is correct according to the input
        x_prime = x
        if x.shape[1] != 512:
            x_prime = torch.cat((x, torch.zeros(64, 512 - x.shape[1]).to('cuda')), dim=1)
            
        encoded = self.encoder(x_prime)
        decoded = self.decoder(encoded)

        #check if decoded shape is correct according to the output, if the output is not 512, then cut the output to the original size of the input
        if decoded.shape[1] != x.shape[1]:
            #print("decoded shape is not correct, cutting the output to the original size of the input")
            decoded = decoded[:, :x.shape[1]]
        return decoded




'''
# Initialize the encoder, decoder, and autoencoder
encoder = Encoder()
decoder = Decoder()
autoencoder = Autoencoder(encoder, decoder)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Assuming you have data for training (visual features)
# visual_features is assumed to be a PyTorch tensor
# Make sure to load and preprocess your data accordingly

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed_visual_features = autoencoder(visual_features)
    
    # Compute loss
    loss = criterion(reconstructed_visual_features, visual_features)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Once training is complete, the 'autoencoder' can be used to map visual features to audio features.

'''