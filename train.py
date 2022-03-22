'''
the code for training WGAN-GP on 2D BSE images

'''
import os
from glob import glob
import argparse
import numpy as np
import matplotlib .pyplot as plt
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error
import tifffile
from tqdm import tqdm
import textwrap

import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

# importing python codes
from functions import crop_images, weights_init, get_noise, get_gradient, gradient_penalty, get_gen_loss, get_crit_loss
from functions import calculate_two_point_df, save_ckp
from Networks import Generator2D,Critic2D 


#parse input arguments

description = textwrap.dedent('''\
    Script to Train a WGAN-GP to reconstruct 2D BSE images.
    ''')

parser = argparse.ArgumentParser()
parser.add_argument( "-ip", "--InputPath", required =  True, type = str, help= textwrap.dedent(''' \
                    Path to original large segmented image of the rock.
                   ''') )
parser.add_argument( "-project_name", "--ProjectName", required =  False, default= 'MetaIgneous', type = str, help= textwrap.dedent(''' \
                    Project name.
                   ''') )
parser.add_argument( "-isize", "--ImageSize", required =  False, default= 128, type = int, help= textwrap.dedent(''' \
                    Training image size.
                   ''') )
args = parser.parse_args()

# parameters
input_path = args.InputPath
project_name = args.ProjectName
isize = args.ImageSize # training image size. this will be used to generate images for training and also fo Generator and discriminator's architecture
n_epochs =  200
display_step = 200
s2_step = 200
batch_size = 128
z_dim = 512 # dimension of noise vector
lr = 0.0001 # learning  rate
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5

im_chan = 1 # number of channels
ngf = 64 # number of generator's filters
ndf = 64 # number of discriminator's filters

torch.manual_seed(0) 
print(input_path)
if __name__ == "__main__":
    
    img = tifffile.imread(input_path)

    print(img.shape)
    ###--------------------------------------------- generate training images
    print('Genearing the training images...')
    training_images_path = os.path.normpath(os.getcwd()) + f'\{project_name}' + r'\Training_images'
    print(training_images_path)
    os.makedirs(training_images_path, exist_ok = True)

    crop_images(img, edge_length = 512, stride = 256, target_path = training_images_path + '/', target_size = isize)
    
    print('training images are ready!')
    print('-----------------------------------')
    
    
    ###---------------------------------------------------------
    # creating dataloader 
    #DATA_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir) + f'\{project_name}'
    
    DATA_DIR = os.path.normpath(os.getcwd()) + f'\{project_name}'
    train_ds = ImageFolder(DATA_DIR, transform= transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                    transforms.ToTensor(),
                                                                   transforms.Normalize( (0.5,), (0.5,) )]))

    train_dl = DataLoader(train_ds, batch_size, shuffle= True, num_workers= 0, pin_memory= True)

    ###-----------------------------------------------------------------------------------
    print('Creating networks...')
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    print (f'Available devices (Number of GPUs): {ngpu} ')
    gen = Generator2D(isize= isize, z_dim= z_dim, ngpu= ngpu, n_extra_layers= 1).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr = lr, betas = (beta_1, beta_2))

    crit = Critic2D(isize = isize, ngpu = ngpu).to(device)
    crit_opt = torch.optim.Adam(crit.parameters(), lr = lr, betas = (beta_1, beta_2))

    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    print('-------------------------------')
    ##---------------------training loop
    print('starting training loop...')
    checkpoint_path = os.path.normpath(os.getcwd()) + '\checkpoints'
    print(f'Checkpoint path: {checkpoint_path}')
    best_model_path = os.path.normpath(os.getcwd()) + '\checkpoints' + r'\best_model'
    print(f'Best model path: {best_model_path}')
    os.makedirs(best_model_path, exist_ok = True)
    
    output_curves = os.path.normpath(os.getcwd()) + '\outputs'
    os.makedirs(output_curves, exist_ok = True)
    print(f'Output curves path : {output_curves}')
    
    mse_s2_min = 1e-5 # just a value to compare mse_s2 in each step and save the best model.
    #mse_s2_min = 50
    #----------------------------------------Training-----------------------------------------------
    cur_step = 0
    generator_losses = []
    critic_losses = []
    
    for epoch in range(n_epochs):
        
        # Dataloader returns the batches
        for real, _ in tqdm(train_dl):
            cur_batch_size = len(real)
            real = real.to(device)

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
    #         crit_sched.step()
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            generator_losses += [gen_loss.item()]

            ###----------------------------- Visualization code------------------------------------- ###
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")


            if cur_step % s2_step == 0 and cur_step >0:

                # convert real and fake tensors to numpy
                fake_np = (fake.detach().cpu().numpy()[:,0,:, :])
                real_np = (real.detach().cpu().numpy()[:,0,:, :])

                ##---------------------------------thresholding images-------------------

                thresh_real = threshold_otsu(real_np) # threshold value from Otsu
                real_thresh = np.where(real_np>= thresh_real, 1, 0).astype(np.uint8) # thresholded real images

                thresh_fake = threshold_otsu(fake_np)
                fake_thresh = np.where(fake_np>= thresh_fake, 1, 0).astype(np.uint8)

                # -----------------------------------for s2 and fn-----------------------------------
                print('Calculating two point correlation...')
                df_s2_real, df_fn_real = calculate_two_point_df(real_thresh)
                df_s2_fake, df_fn_fake = calculate_two_point_df(fake_thresh)
                print('Dataframes created!')

                #-------------------Two point correlations
                
                #scaled s2 (fn)---
                plt.figure()
                plt.plot(df_fn_real.index, df_fn_real['fn']['mean'], color ='b', label = 'Real')
                plt.plot(df_fn_fake.index, df_fn_fake['fn']['mean'], color = 'r', label = 'Fake' )
                plt.xlabel('r(px)', fontsize = 'x-large')
                plt.ylabel('$F_n$', fontsize = 'x-large')
                plt.legend(fontsize='x-large')
                mse_fn = mean_squared_error(df_fn_real['fn']['mean'], df_fn_fake['fn']['mean'])
                plt.title(f'MSE={mse_fn}')
                plt.grid()
                plt.savefig(output_curves + '/fn' + "_" + str(cur_step) + '.png', dpi = 1500)
                #plt.show()

                #s2---           
                plt.figure()
                plt.plot(df_s2_real.index, df_s2_real['s2']['mean'], color ='b', label='Real')
                plt.plot(df_s2_fake.index, df_s2_fake['s2']['mean'], color = 'r', label='Fake' )
                plt.xlabel('r(px)', fontsize = 'x-large')
                plt.ylabel('$S_2$', fontsize = 'x-large')
                plt.legend(fontsize ='x-large')

                mse_s2 = mean_squared_error(df_s2_real['s2']['mean'], df_s2_fake['s2']['mean'])
                plt.title(f'MSE={mse_s2}')
                plt.grid()
                plt.savefig(output_curves + '/s2' + "_" + str(cur_step) + '.png', dpi = 1500)
                #plt.show()


                if mse_s2 <= 5e-7: #5e-6:

                    # create checkpoint variable and add important data
                    checkpoint = {
                        'step':cur_step,
                        's2_min': mse_s2,
                        'gen_state_dict': gen.state_dict(),
                        'gen_optimizer': gen_opt.state_dict(),
                        'crit_state_dict': crit.state_dict(),
                        'crit_optimizer': crit_opt.state_dict()
                    }


                    # save checkpoint
                    save_ckp(checkpoint, False, checkpoint_path=checkpoint_path + '/WGANGP_' + str(cur_step) + '.pt',
                             best_model_path= best_model_path +'/WGANGP_' + str(cur_step) + '.pt')

                    # save the model if mse_s2 has decreased

                    if mse_s2 <= mse_s2_min:
                        print('MSE_S2 decreased ({:.9f} --> {:.9f}). saving model...'.format(mse_s2_min, mse_s2))

                        #removing the previous best model in the folder cause they're not the best anymore
                        for filename in glob(best_model_path + '/WGAN*'):

                            os.remove(filename) 
                        # save the checkpoint as the best model
                        save_ckp(checkpoint, True, checkpoint_path= checkpoint_path + '/WGANGP_' + str(cur_step) + '.pt',
                                 best_model_path= best_model_path +'/WGANGP_' + str(cur_step) + '.pt')

                        mse_s2_min = mse_s2       

            cur_step += 1