import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_with_steps(model, scheduler, train_config, model_config, diffusion_config, num_images, output_dir):
    xt = torch.randn((1,
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    
    timesteps = list(range(diffusion_config['num_timesteps']))
    save_steps = [timesteps[i * (diffusion_config['num_timesteps'] // (num_images - 1))] for i in range(num_images - 1)]
    save_steps.append(timesteps[-1])  # Ensure the last step is included

    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        if i in save_steps:
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=1)
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(os.path.join(output_dir, f'x0_step_{i}.png'))
            img.close()

def main(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Carregar o checkpoint da última época
    checkpoint_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample_with_steps(model, scheduler, train_config, model_config, diffusion_config, args.num_images, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample images with steps for DDPM')
    parser.add_argument('--config', dest='config_path', default='config/default.yaml', type=str)
    parser.add_argument('--num_images', type=int, default=10, help='Number of intermediate images to save')
    parser.add_argument('--output_dir', type=str, default='mysample', help='Directory to save the output images')
    args = parser.parse_args()
    main(args)
