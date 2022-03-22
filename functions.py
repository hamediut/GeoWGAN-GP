import numpy as np
import numba
import pandas as pd
from numba import jit
from skimage.transform import resize
import tifffile
from tqdm import tqdm


#pytorch library
import torch
from torch import nn
#from tqdm.auto import tqdm
#import torchvision
#from torchvision import transforms
#from torchvision.utils import make_grid



def crop_images(img, edge_length, stride, target_path, target_size = None):
    M=N= edge_length
    I_inc = J_inc = stride
    
    count = 0
    for i in tqdm(range(0, img.shape[0], I_inc)):
        for j in range(0, img.shape[1], J_inc):

            if len(img.shape) == 3:# check if the image is color image , channels = 3
                subset = img[i:i+N, j:j+N, :]
                if subset.shape == (N, M, 3):
                    tifffile.imsave(target_path + 'crop' +"_"+str(count)+ '.tif', subset)
                    count += 1
    
                
            elif len(img.shape) == 2:
                subset = img[i:i+N, j:j+N]
                if subset.shape == (N, M):
                    resized_subset = subset.astype('uint8')
                    if target_size is not None:
                        resized_subset = (resize(subset, (target_size, target_size), mode= 'constant', preserve_range= True,
                      anti_aliasing= True, anti_aliasing_sigma= False))
                        
                        # sometimes resizing creates artifacts and intermediate values between 0 and 255
                        resized_subset =  np.where(resized_subset > 0, 255, 0).astype('uint8')
                        
                    tifffile.imsave(target_path + 'crop' +"_"+str(count)+ '.tif', resized_subset)

                    count += 1
    print(f'Number of cropped images= {count}')

@jit 
# --> It is preferred to use numba here for a speed-up, if installed!!
def two_point_correlation(im, dim, var=0):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 0)
    
    The input image im is expected to be three-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[1] #y-axis
        dim_2 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2-r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a+r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a+r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r]/(float(lmax))
    return two_point

def calculate_two_point_df(images):
    """
    This function calculates average two-point correlations (s2 and fn) from images and convert them to dataframe.
    """
    
    s2_list = []
    fn_list = []
    
    for i in range(images.shape[0]):
    
        two_pt_dim0 = two_point_correlation(images[i], dim = 0, var = 1) #S2 in x-direction
        two_pt_dim1 = two_point_correlation(images[i], dim = 1, var = 1) #S2 in y-direction

        #Take average of directions; use half linear size assuming equal dimension sizes
        Nr = two_pt_dim0.shape[0]//2

        S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
        S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
        S2_average = ((S2_x + S2_y)/2)[:Nr]
        
        s2_list.append(S2_average)
        
        # autoscaled covriance---------------------------------------
        f_average = (S2_average - S2_average[0]**2)/S2_average[0]/(1 - S2_average[0])
        fn_list.append(f_average)
    
    # from list to dataframe----------
    
    df_list = []
    for i in np.arange(0, len(s2_list)):
        df_list.append(pd.DataFrame(s2_list[i], columns = ['s2'] ) )
    df = pd.concat(df_list)
    df['r'] = df.index
    df_grouped = df.groupby( ['r'] ).agg( {'s2': [np.mean, np.std, np.size] } )
    
    
    df_fn_list = []
    for i in np.arange(0, len(fn_list)):
        df_fn_list.append(pd.DataFrame(fn_list[i], columns = ['fn'] ) )
    df_fn = pd.concat(df_fn_list)
    df_fn['r'] = df_fn.index
    df_fn_grouped = df_fn.groupby( ['r'] ).agg( {'fn': [np.mean, np.std, np.size] } )
        
        
    return df_grouped, df_fn_grouped


###-----------------------------------------------------------------save and load checkpoints----------------------------------------------
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save. state is a dictionary containing:
    epoch, S2_min, model.state_dict, optimizer.state_dict.
    
    is_best: is the best checkpoint: True or False
    checkpoint_path: path to save checkpoint
    best_model_path: path to save the best model
    
    """   
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)
        

def load_cpk(checkpoint_fpath, gen, gen_opt, crit, crit_opt, device='cpu'):
    """
    
    checkpoint_path: path to load checkpoint from
    gen: an instance of generator that we want to load the state (what we've saved) into
    gen_opt: generator's optimizer we defined in previous training
    crit: an instance of critic that we want to load the state (what we've saved) into
    crit_opt: critic's optimizer we defined in previous training
    
    device: the device to load the trained model on. For inference, use cpu;otherwise you get an cuda out of memory error
    returns:
    gen, gen_opt, crit, crit_opt, step, s2_min
    """
    
    #load checkpoint
    checkpoint = torch.load(checkpoint_fpath, map_location= torch.device(device))
    
    #Generator----------------------------
    # initialize state_dict from checkpoint to generator:
    gen.load_state_dict(checkpoint['gen_state_dict'])
    
    # initialize optimizer from checkpoint to generator'optimizer
    gen_opt.load_state_dict(checkpoint['gen_optimizer'])
    
    #Critic-------------------------------
    crit.load_state_dict(checkpoint['crit_state_dict'])
    crit_opt.load_state_dict(checkpoint['crit_optimizer'])
    
    
    # initialize s2_min from checkpoint
    s2_min = checkpoint['s2_min']
    
    
    return gen, gen_opt, crit, crit_opt, checkpoint['step'], s2_min


def show_tensor_images(image_tensor, num_images = 16, size = (1, 128, 128), nrows = 4, save_plot = False, out_format = 'png' ,output_path = None, file_name = None):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size of image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrows)
    style.use('default')
    plt.figure()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    if save_plot:
        plt.savefig(output_path + file_name + '.' + out_format, format = out_format, dpi = 1500)
    plt.show()


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
##-------------------------------------------------------W loss and gradient penalty--------------------
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together: this is the x_hat in the lecture
    mixed_images = real * epsilon + fake * (1 - epsilon) 

    # Calculate the critic's scores on the mixed images: C(x_hat)
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images: gradient of c(x_hat)
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad

        inputs=mixed_images,
        outputs=mixed_scores,

        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

#----------------
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)   
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean( ((gradient_norm - 1 )**2 ) )

    return penalty
##---------------------
def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''

    gen_loss = -torch.mean(crit_fake_pred)

    return gen_loss

#-----------------------------
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''

    crit_loss = torch.mean(crit_fake_pred - crit_real_pred) + (c_lambda * gp)

    return crit_loss