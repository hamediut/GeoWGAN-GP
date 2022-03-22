#pytorch library
import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from torch.utils.data.dataloader import DataLoader


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator2D(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, isize= 128, z_dim = 512, im_chan = 1, ngf = 64, ngpu= 1, n_extra_layers = 1):
        super(Generator2D, self).__init__()
        self.z_dim = z_dim
        self.ngpu = ngpu
        
        assert isize % 16 == 0  # isize has to be a multiple of 16
        
        # this is to find the first output channel that we need. if we start with this number,
        # we end up with the desired image size at the end. for example for an output shape of 128 ^3, cngf will be 1024.
        # the we halve this at each step.
        
        cngf, tisize = ngf//2, 4 
        while tisize != isize:
            cngf = cngf * 2       # 64, 128, 256, 512, 1024, 2048
            tisize = tisize * 2   # 8,  16,  32,  64,  128,  256 = isize (desired output)
            
        #here we define the first TransposeConv layer---------------------------------------------   
        # input size is: batch_size, im_chan, image_size ^3 =  batch_size, z_dim=512, 1^3
        main = nn.Sequential(
            # input is Z, going into a Transposed convolution: from 512 --> 1024
            nn.ConvTranspose2d(in_channels = z_dim, out_channels = cngf, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(cngf),
            nn.ReLU(inplace = True)        
                
        )
         # output size= 1, 2048, 4^2
        
        # add more TransposeConv up to one layer before the last where we have a Conv layer to remove checkerboard pattern artifacts        
        i, csize, cndf = 3, 4, cngf # i is the layer number
                        
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2
        
        # i=15 (for isize= 256^3) after the last ConvTrans layer
        #cngf = 64
        # output size = 1, 64, 128^3
        
        # Extra layers: Cov layer before last layer
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv2d(cngf, cngf, kernel_size = 1, stride = 1, padding = 0, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm2d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
        # i=17 
        
        main.add_module(str(i),
                        nn.ConvTranspose2d(cngf, im_chan, kernel_size = 4, stride = 2, padding = 1, bias=False))
        main.add_module(str(i+1), nn.Tanh() )
        
        self.main = main
        
        #--------------------------------------------------------------------------------------------     
        
    def unsqueeze_noise(self, noise):
        
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)
        
        
        
#     def forward(self, noise): # for a single gpu
        
#         '''
#         Function for completing a forward pass of the generator: Given a noise tensor, 
#         returns generated images.
#         Parameters:
#             noise: a noise tensor with dimensions (n_samples, z_dim)
#         '''
#         x = self.unsqueeze_noise(noise)
#         return self.main(x)
    
    def forward(self, noise):# for multiple gpus
        
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        gpu_ids = None 
        if isinstance(noise, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)
    
#-------------------------------------------------------------

