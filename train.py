

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


# Training Loop

def fit(model, train_dl, device, latent_size, fixed_latent, save_samples, criterion, epochs, lr, start_idx=1): #clipvalue=1.0,
    # set models to a training mode
    model["discriminator"].train()
    model["generator"].train()
    torch.cuda.empty_cache() # clear GPU cache
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    optimizer = {
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(), 
        lr=lr, betas=(0.5,0.999)), 
        "generator": torch.optim.Adam(model["generator"].parameters(), 
        lr=lr, betas=(0.5,0.999))
    }
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer["discriminator"], step_size=50, gamma=0.5)
    
    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []
        for real_images, _ in tqdm(train_dl):

            # train discriminator
            real_images = real_images.to(device)
            cur_batch_size = real_images.size(0)
            optimizer["discriminator"].zero_grad() # Clear discriminator gradients

            # pass real images through discriminator
            real_preds = model["discriminator"](real_images) # discriminator forward-pass with real images as input
            real_targets = torch.ones(cur_batch_size, real_preds.size(1), device=device) # generate real targets
            real_loss = criterion(real_preds, real_targets) # calculate the discriminator loss
            cur_real_score = torch.mean(real_preds).item() # calculate the current real score
            
            # generate fake images
            latent = torch.randn(cur_batch_size, latent_size,
                                 1, 1, device=device) # generate latents of
                                                                               # current batch size x latent_size
            fake_images = model["generator"](latent) # generator forward-pass

            # pass fake images through discriminator
            fake_preds = model["discriminator"](fake_images) # discriminator forward-pass with fake images as input
            fake_targets = torch.zeros(cur_batch_size, fake_preds.size(1), device=device) # generate fake targets
            fake_loss = criterion(fake_preds, fake_targets) # calculate the discriminator loss
            cur_fake_score = torch.mean(fake_preds).item() # calculate the current fake score

            # save scores
            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)

            # Update discriminator weights
            loss_d = real_loss + fake_loss # calculate the overall discriminator loss
            loss_d.backward() # discriminator backward-pass
            optimizer["discriminator"].step() # update discriminator weights
            loss_d_per_epoch.append(loss_d.item()) # save the loss value

            # train generator
            optimizer["generator"].zero_grad() # clear generator gradients
            
            # generate fake images
            latent = torch.randn(cur_batch_size, latent_size, 1,
                                 1, device=device) # generate latents of# current batch_size x latent_size
            fake_images = model["generator"](latent) # generator forward-pass with latents
            
            # try to fool the discriminator
            preds = model["discriminator"](fake_images) # discriminator forward-pass with fake generated images
            targets = torch.ones(cur_batch_size, preds.size(1), device=device)
            loss_g = criterion(preds, targets) # calculate the generator loss
            
            # Update generator weights
            loss_g.backward() # generator backward-pass
            optimizer["generator"].step() # update generator weights
            loss_g_per_epoch.append(loss_g.item()) # save the loss value
        
        # save losses & scores per epoch
        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))
        real_scores.append(np.mean(real_score_per_epoch))
        fake_scores.append(np.mean(fake_score_per_epoch))

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, 
            losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent) 
        torch.save(model["discriminator"].state_dict(), '/content/drive/MyDrive/discriminator_model_150_epochs.pt')
        torch.save(model["generator"].state_dict(), '/content/drive/MyDrive/generator_model_150_epochs.pt')
    return losses_g, losses_d, real_scores, fake_scores
