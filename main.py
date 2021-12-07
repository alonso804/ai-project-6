import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math

def Show(out, title = ''):
  print(title)
  out = out.permute(1,0,2,3)
  grilla = torchvision.utils.make_grid(out,10,5)
  plt.imshow(transforms.ToPILImage()(grilla), 'jet')
  plt.show()

def Show_Weight(out):
  grilla = torchvision.utils.make_grid(out)
  plt.imshow(transform.ToPILImage()(grilla), 'jet')
  plt.show()

def train(model, train_loader, Epochs, loss_fn):
    train_loss_avg = []
    for epoch in range(Epochs):
      train_loss_avg.append(0)
      num_batches = 0
    
      for image_batch, _ in train_loader:
          image_batch = image_batch.to(device)
        #  print(image_batch.size())
          
          image_batch_recon = model(image_batch)
         # print(image_batch_recon.size())
          loss = loss_fn(image_batch_recon, image_batch)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          
      train_loss_avg[-1] /= num_batches
      print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, Epochs, train_loss_avg[-1]))
    return train_loss_avg


def main():
    batch_size = 64

    img_transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5,),(0.5,))]) 

    train_set = torchvision.datasets.MNIST(root = '../../data', train= True, transform= img_transform, download= True)
    test_set = torchvision.datasets.MNIST(root = '../../data', train= False, transform= img_transform, download= True)

    img, _ = train_set[0]
    print(img.shape)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    
    
    capacity = 64
    latent_dims = 10    
    learning_rate = 0.001
    autoencoder = Autoencoder()
    autoencoder.to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    autoencoder.train()


    loss_result = train(autoencoder,train_loader,20,loss)