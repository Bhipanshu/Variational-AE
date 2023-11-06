import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
from torchvision import transforms , datasets

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

#VARIATIONAL AUTOENCODERS

# Height: (original_height + 2*padding - kernel_size) / stride + 1
# Width: (original_width + 2*padding - kernel_size) / stride + 1

#writer summary
from torch.utils.tensorboard import SummaryWriter


#defining the dataloader
transform1 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
    ])

dataset = datasets.ImageFolder(root='PATH TO THE DATASET',transform=transform1)
dataloader = torch.utils.data.DataLoader(dataset,batch_size= 40,shuffle=True)


#defining the model
class VariationalEncoders(nn.Module):
    def __init__(self,latent_dims):
        super(VariationalEncoders,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear1 = nn.Linear(128*32*32,64)
        self.linear2 = nn.Linear(64,latent_dims)
        self.linear3 = nn.Linear(64,latent_dims)
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self,x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        x = x.view(x.size(0),-1)

        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        logvar = self.linear3(x)

        return mu, logvar

    
class VariationalDecoder(nn.Module):
    def __init__(self,latent_dims):
        super(VariationalDecoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,latent_dims*2)
        self.linear2 = nn.Linear(latent_dims*2,128*32*32)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2,padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2,padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=2,stride=2,padding=0)

    def forward(self,z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = z.view(z.size(0),128,32,32)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z)) 
        return z



class VariationalAutoEncoders(nn.Module):
    def __init__(self,latent_dims):
        super(VariationalAutoEncoders,self).__init__()
        self.encoder = VariationalEncoders(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self,x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, self.decoder(z)


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

criterion = nn.MSELoss()

def loss_function(x, x_hat, mu, logvar):

    criterion = nn.MSELoss()
    mse_loss = criterion(x,x_hat)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    return mse_loss + kld_loss*0.00025





def train(vautoencoder, dataloader, model_checkpoint_path, epochs=60,writer=None):
    opt = torch.optim.Adam(vautoencoder.parameters(), lr=1e-4)
    vautoencoder.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):

            x = x.to(device) # GPU
            mu, sigma, x_hat = vautoencoder(x)
            loss = loss_function(x, x_hat, mu, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step = epoch * len(dataloader) + batch_idx  # Calculate global step
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            print(f'epoch {epoch}/{epochs} batch_idx {batch_idx}/{len(dataloader)} Loss : {loss.item():.4f}', end='\r')
        
        print(f'epoch {epoch}/{epochs} Loss : {loss.item():.4f}')
        torch.save(vautoencoder.state_dict(), model_checkpoint_path)
        
    return vautoencoder

def test(vautoencoder):
    vautoencoder.eval()

    with torch.no_grad():
        z = vautoencoder.reparameterize(torch.randn(1,1024).to(device),torch.randn(1,1024).to(device))
        output = vautoencoder.decoder(z)

    return output




# import matplotlib.pyplot as plt
# def plot_latent(autoencoder,data,num_batches=100):
#     for i,(x,y) in enumerate(data):
#         z = autoencoder.encoder(x.to(device))
#         z = tuple(tensor.to('cpu').detach().numpy() for tensor in z)
#         # z = z.to('cpu').detach().numpy()
#         # print(z)
#         plt.scatter(z[0][:,0],z[0][:,1],c=y,cmap='tab10')
#         if i>num_batches:
#             plt.colorbar()
#             break



def main():
    #model Params
    latent_dims = 1024
    model_checkpoint_path = 'vae_model.pt'  # Specify the model checkpoint file path

    log_dir = "logs"  # Specify your desired log directory
    writer = SummaryWriter(log_dir)

    #calling the model
    vae = VariationalAutoEncoders(latent_dims).to(device)
    vae.load_state_dict(torch.load(model_checkpoint_path))

    #training the model 
    vae = train(vae,dataloader, model_checkpoint_path=model_checkpoint_path, writer = writer)

    from PIL import Image
    ima = test(vae)
    output_image = transforms.ToPILImage()(ima[0])
    output_image.save('test-vae_1.png')
    output_image.show()
    
    # transform = transforms.Compose([
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor()
    # ])
    # latent_vector = torch.randn(1,latent_dims).to(device)
    # print(latent_vector.shape)
    # with torch.no_grad:
    #     output = vae(latent_vector)

    # output = output.cpu().numpy()
    # output = np.transpose(output, (0, 2, 3, 1))
    # plt.imshow(output[0])
    # plt.show()
    # plot_latent(vae,dataloader)

    print("done")


if __name__ == '__main__':
    main()

