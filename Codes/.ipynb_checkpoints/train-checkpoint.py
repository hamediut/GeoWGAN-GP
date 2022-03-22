'''
the code for training WGAN-GP on 2D BSE images

'''
import os
import argparse
import numpy as np
from skimage.transform import resize
import tifffile
from tqdm import tqdm
import textwrap

#import torchvision
#from torchvision.datasets import ImageFolder
#from torch.utils.data.dataloader import DataLoader

# importing python codes
from functions import crop_images
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
args = parser.parse_args()

# parameters
input_path = args.InputPath
project_name = args.ProjectName
isize = 128 # training image size. this will be used to generate images for training and also fo Generator and discriminator's architecture
batch_size = 128


print(input_path)
img = tifffile.imread(input_path)

print(img.shape)
###--------------------------------------------- generate training images
print('Genearing the training images...')
training_images_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + f'\{project_name}' + r'\Training_images'
print(training_images_path)
#os.makedirs(training_images_path, exist_ok = True)

#crop_images(img, edge_length = 512, stride = 256, target_path = training_images_path + '/', target_size = isize)

###---------------------------------------------------------
# creating dataloader 
DATA_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir) + f'\{project_name}'
train_ds = ImageFolder(DATA_DIR, transform= transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                transforms.ToTensor(),
                                                               transforms.Normalize( (0.5,), (0.5,) )]))

train_dl = DataLoader(train_ds, batch_size, shuffle= True, num_workers= 4, pin_memory= True)

# # we select 64 random images from real images to calculate avergae auoscaled s2 
# # and use it as input for simulated annnealing
real_image, _ = next(iter(train_dl))
print('Image shape:{}'.format(real_image.shape))
#convert to numpy
real_image_np = (real_image.detach().cpu().numpy()[:, 0, :, :]) #(batch_size, channel, w, h)
print(f'Numpy image shape : {real_image_np.shape}')

