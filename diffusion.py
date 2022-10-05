!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update



!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2



!nvcc --version


!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git


%load_ext nvcc_plugin


pip install denoising_diffusion_pytorch

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
import torchvision.transforms as transforms

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()



diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 100,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()


from google.colab import drive
drive.mount('/content/drive')


trainer = Trainer(diffusion, './drive/MyDrive/fundus', 
                  train_batch_size = 4, 
                  train_lr= 2e-5, 
                  train_num_steps = 10000, 
                  gradient_accumulate_every = 2, 
                  ema_decay = 0.995, 
                  amp = True)

trainer.train()

sampled_images = diffusion.sample(batch_size = 4)
tf = transforms.ToPILImage()
img_1 = tf(sampled_images[1])
img_2 = tf(sampled_images[2])
img_3 = tf(sampled_images[3])
img_4 = tf(sampled_images[4])

img_1

img_2

img_3

img_4

