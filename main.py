import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import GeneratorNet, DiscriminatorNet
import matplotlib.pyplot as plt


## TODO convert to DCGAN https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

    # Noise
def noise(size):
    return torch.randn(size, 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
# Number of epochs


def real_data_target(size):
    return torch.ones(size, 1).to(device)   

def fake_data_target(size):
    return torch.zeros(size, 1).to(device)
    

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

data = datasets.MNIST(root='data', train=True, 
   transform=transforms.Compose([transforms.ToTensor()]), download=True)
loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    print('epoch {}'.format(epoch))
    for n_batch, (real_batch,_) in enumerate(loader):
        # 1. Train Discriminator
        real_data = images_to_vectors(real_batch).to(device)
        
        # Generate fake data
        fake_data = generator(noise(real_data.size(0)).to(device))

        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(
           d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)).to(device))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log error

img = generator(noise(1).to(device)).detach().cpu().numpy()