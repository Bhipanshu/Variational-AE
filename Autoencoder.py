import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms , datasets
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#defining the dataloader
transform1 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='Path to data dir',transform=transform1)
dataloader = torch.utils.data.DataLoader(dataset,batch_size= 40,shuffle=True)


#defining the model
#ENCODER
class Encoders(nn.Module):
    def __init__(self):
        super(Encoders,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        return x


#DECODER
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2,padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2,padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=2,stride=2,padding=0)

    def forward(self,z):
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z)) 
        return z


class AutoEncoders(nn.Module):
    def __init__(self):
        super(AutoEncoders,self).__init__()
        self.encoder = Encoders()
        self.decoder = Decoder()

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)
    

criterion = nn.MSELoss()
def train(autoencoder, dataloader, model_checkpoint_path, epochs=50,writer=None):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    for epoch in range(epochs):
        kl = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device) # GPU
            x_hat = autoencoder(x)
            loss = criterion(x_hat,x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step = epoch * len(dataloader) + batch_idx  # Calculate global step
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            print(f'epoch {epoch}/{epochs} batch_idx {batch_idx}/{len(dataloader)} Loss : {loss.item():.4f}', end='\r')
        
        print(f'epoch {epoch}/{epochs} Loss : {loss.item():.4f}')
        torch.save(autoencoder.state_dict(), model_checkpoint_path)
        
    return autoencoder





def main():
    model_checkpoint_path = './newautoencoder.pt'  # Specify the model checkpoint file path

    log_dir = "logs"  # Specify your desired log directory
    writer = SummaryWriter(log_dir)

    #calling the model
    vae = AutoEncoders().to(device)

    #training the model 
    vae = train(vae,dataloader, model_checkpoint_path=model_checkpoint_path, writer = writer)

    from PIL import Image


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    image_path = 'Path to single image'


    image = Image.open(image_path)


    image_tensor = transform(image).unsqueeze(0)
    autoencoder = AutoEncoders()
    autoencoder.load_state_dict(torch.load(model_checkpoint_path))

    with torch.no_grad():
        output = autoencoder(image_tensor.to(device).cpu())


    output_image = transforms.ToPILImage()(output[0])
    output_image.save('test.png')
    output_image.show()


if __name__ == '__main__':
    main()

