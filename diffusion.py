%%%
% We modified the code from https://github.com/lucidrains/denoising-diffusion-pytorch to provide a 
% hand-on experience to generate fundus photographs using DDPM.
% -Ho, J., Jain, A. & Abbeel, P. Denoising diffusion probabilistic models. in Proceedings of the 34th 
% International Conference on Neural Information Processing Systems 6840–6851 (2020)
% The training with large resolution images (more than 64 x 64 pixels) needs a lot of time. Early stop 
% of training will provide some weird images.
%%%

% This code is designed for implementation in the Colab.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from google.colab import drive
drive.mount('/content/drive')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()


trainer = Trainer(diffusion, './drive/MyDrive/fundus', 
                  train_batch_size = 16, 
                  train_lr= 2e-5, 
                  train_num_steps = 1000000, 
                  gradient_accumulate_every = 2, 
                  ema_decay = 0.995, 
                  amp = True)

trainer.train()

sampled_images = diffusion.sample(batch_size = 4)
tf = transforms.ToPILImage()
img_1 = tf(sampled_images[1])

%% show the image
img_1
