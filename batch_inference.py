"""
Batch Sampling Scripts of LightningDiT.
"""

import os, math, json, pickle, logging, argparse, yaml, torch, numpy as np
import gc
import torch.nn.functional as F
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision
try:
    import lpips
except ImportError:
    lpips = None


def variable_size_collate_fn(batch):
    """Collate samples with variable spatial sizes by resizing to 720x720.
    
    All images are resized to 720x720 (multiple of 16) for processing,
    and original pixel sizes are saved for later resizing back.
    
    Returns the standard 5-tuple PLUS:
    - original_latent_sizes: for latent dimension tracking
    - original_pixel_sizes: for resizing output images back
    """
    VAE_DOWNSAMPLE = 16
    TARGET_SIZE = 720  # Resize all images to 720x720
    
    latent_gts, latent_ins, pixel_gts, pixel_ins, labels = zip(*batch)
    
    # Save original pixel sizes before resizing
    original_pixel_sizes = [
        (t.shape[-2], t.shape[-1])  # (H, W) in pixels
        for t in pixel_ins
    ]
    
    # Save original latent sizes (derived from original pixel sizes)
    original_latent_sizes = [
        (h // VAE_DOWNSAMPLE, w // VAE_DOWNSAMPLE)
        for h, w in original_pixel_sizes
    ]
    
    # Resize all images to TARGET_SIZE x TARGET_SIZE
    # Convert tensor to PIL, resize, then convert back to tensor
    def resize_tensor_to_720(t):
        """Resize (C, H, W) tensor to (C, 720, 720)"""
        # Convert to PIL: tensor (C, H, W) -> numpy (H, W, C) -> PIL
        # First denormalize from [-1, 1] to [0, 1]
        t_denorm = (t + 1.0) / 2.0
        t_denorm = torch.clamp(t_denorm, 0, 1)
        # Permute to (H, W, C) and convert to numpy
        t_np = t_denorm.permute(1, 2, 0).cpu().numpy()
        t_np = (t_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(t_np)
        
        # Resize to 720x720
        pil_img = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        
        # Convert back to tensor: PIL -> numpy -> tensor (C, H, W)
        t_np = np.array(pil_img).astype(np.float32) / 255.0
        t_tensor = torch.from_numpy(t_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        # Renormalize to [-1, 1]
        t_tensor = t_tensor * 2.0 - 1.0
        return t_tensor
    
    # Resize all pixel images to 720x720
    resized_pixel_gts = [resize_tensor_to_720(t) for t in pixel_gts]
    resized_pixel_ins = [resize_tensor_to_720(t) for t in pixel_ins]
    
    # Stack resized images (all are now 720x720, so no padding needed)
    padded_pix_gts = torch.stack(resized_pixel_gts)  # (N, 3, 720, 720)
    padded_pix_ins = torch.stack(resized_pixel_ins)  # (N, 3, 720, 720)
    stacked_labels = torch.stack(list(labels))
    
    # Latent dimensions after resize (720 // 16 = 45)
    target_lat_h = TARGET_SIZE // VAE_DOWNSAMPLE
    target_lat_w = TARGET_SIZE // VAE_DOWNSAMPLE
    
    # Pad latents to target size (if needed, though they should already match)
    def pad2d(t, target_h, target_w):
        """Pad (C, H, W) tensor to (C, target_h, target_w) with zeros on bottom/right."""
        return F.pad(t, (0, target_w - t.shape[-1], 0, target_h - t.shape[-2]))
    
    padded_lat_gts = torch.stack([pad2d(t, target_lat_h, target_lat_w) for t in latent_gts])
    padded_lat_ins = padded_lat_gts  # unused placeholder
    
    return padded_lat_gts, padded_lat_ins, padded_pix_gts, padded_pix_ins, stacked_labels, original_latent_sizes, original_pixel_sizes

# local imports
from tokenizer.vavae import VA_VAE
from models.lightningdit_addy import LightningDiT_models
from transport import create_transport, Sampler
from datasets.img_latent_dataset import ImgLatentDataset1



DATASETS_DIR = '/openbayes/input/input0/latents_with_origin_data/'
DATASETS_LIST = [
    os.path.join(DATASETS_DIR, a) 
    for a in os.listdir(DATASETS_DIR) 
    if 'train' not in a.lower() and 'val' not in a.lower() and 'GoPro' not in a
]
DATASETS_LIST = [
    '/openbayes/input/input0/latents_with_origin_data/Rain100L_512',
    '/openbayes/input/input0/latents_with_origin_data/BSD68_512',
    '/openbayes/input/input0/latents_with_origin_data/LoL_512',
    '/openbayes/input/input0/latents_with_origin_data/RESIDE_512',
    '/openbayes/input/input0/latents_with_origin_data/GoPro_512',
#     '/openbayes/input/input0/latents_with_original_size/under',
#     '/openbayes/input/input0/latents_with_original_size/LoL',
#     '/openbayes/input/input0/latents_with_original_size/Outdoor-Rain',
]
print(DATASETS_LIST)

def calculate_metrics_for_folder(folder_dir, accelerator, no_fusion=False, lpips_model=None):
    if not accelerator.is_main_process:
        return None, None, None

    # Ensure imports are available (safeguard for distributed execution)
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_func
        from skimage.metrics import structural_similarity as ssim_func
    except ImportError:
        # Fallback if import fails
        raise ImportError("Failed to import skimage.metrics functions. Please ensure scikit-image is installed.")

    if no_fusion:
        # For no fusion, look for files with _nofusion suffix
        gen_files = sorted(glob(os.path.join(folder_dir, "[0-9]*_nofusion.png")))
    else:
        # For fusion, look for files without _nofusion suffix
        gen_files = sorted(glob(os.path.join(folder_dir, "[0-9]*.png")))
        gen_files = [f for f in gen_files if "_" not in os.path.basename(f)]
    
    psnrs, ssims, lpips_scores = [], [], []
    
    # Initialize LPIPS model if not provided and lpips is available
    if lpips_model is None and lpips is not None:
        lpips_model = lpips.LPIPS(net='vgg').eval()
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
    
    for f in gen_files:
        base = f.replace("_nofusion.png", "").replace(".png", "")
        gt_path = f"{base}_gt.png"
        
        if os.path.exists(gt_path):
            img = np.array(Image.open(f))
            gt = np.array(Image.open(gt_path))
            
            psnrs.append(psnr_func(gt, img, data_range=255))
            ssims.append(ssim_func(gt, img, data_range=255, channel_axis=2))
            
            # Calculate LPIPS if available
            if lpips_model is not None:
                # Convert images to tensor: (H, W, C) -> (1, C, H, W), normalize to [-1, 1]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W) in [0, 1]
                gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
                gt_tensor = gt_tensor * 2.0 - 1.0
                img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
                gt_tensor = gt_tensor.unsqueeze(0)
                
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                    gt_tensor = gt_tensor.cuda()
                
                with torch.no_grad():
                    lpips_score = lpips_model(img_tensor, gt_tensor)
                    lpips_scores.append(lpips_score.item())
            
    if len(psnrs) == 0: 
        return 0.0, 0.0, 0.0
    avg_lpips = np.mean(lpips_scores) if len(lpips_scores) > 0 else 0.0
    return np.mean(psnrs), np.mean(ssims), avg_lpips

def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """
    Run sampling.
    """
    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 9.0
        
    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(sample_folder_dir, exist_ok=True) 
    else:
        png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
        png_count = len(png_files)
        # Keep fid_num aligned with the test set size so all samples can be processed.
        if png_count > train_config['sample']['fid_num']:
            if accelerator.process_index == 0:
                print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
            # Initialize LPIPS model once for reuse
            lpips_model = None
            if lpips is not None:
                lpips_model = lpips.LPIPS(net='vgg').eval()
                if torch.cuda.is_available():
                    lpips_model = lpips_model.cuda()
            avg_psnr, avg_ssim, avg_lpips = calculate_metrics_for_folder(sample_folder_dir, accelerator, no_fusion=False, lpips_model=lpips_model)
            avg_psnr_nofusion, avg_ssim_nofusion, avg_lpips_nofusion = calculate_metrics_for_folder(sample_folder_dir, accelerator, no_fusion=True, lpips_model=lpips_model)
            return avg_psnr, avg_ssim, avg_lpips, avg_psnr_nofusion, avg_ssim_nofusion, avg_lpips_nofusion

    torch.backends.cuda.matmul.allow_tf32 = True 
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Model is passed in arguments, skip loading logic here

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    ) 
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    using_cfg = cfg_scale > 1.0

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # Calculation logic for batching
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    
    if rank == 0 and accelerator.process_index == 0:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
            
    # Dataset Setup
    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
        
    dataset = ImgLatentDataset1(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0, is_train=False
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)
    
    per_proc_batch_size = train_config['sample']['per_proc_batch_size']
    num_workers = min(4, train_config['data']['num_workers'])  # cap at 4 for inference
    downsample_ratio = train_config['vae'].get('downsample_ratio', 16)

    # SafetensorError is now wrapped as RuntimeError in __getitem__, so
    # num_workers > 0 is safe (no unpicklable exceptions cross process boundary).
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_size_collate_fn,
        drop_last=False,
    )

    total = 0
    
    # Sampling Loop
    # loader yields: (lat_gt_unused, lat_in_unused, pix_gt, pix_in, label, original_latent_sizes, original_pixel_sizes)
    # lat_in is NO LONGER used — latent is obtained by encoding pix_in on-the-fly.
    for x, y, gts, inputs, label, original_latent_sizes, original_pixel_sizes in tqdm(loader, disable=not accelerator.is_local_main_process):
        # label = label.to(device)

        # ── Encode input pixel images on-the-fly ──────────────────────────────
        # inputs: (N, 3, H_pix, W_pix), range [-1,1], H/W are multiples of downsample_ratio
        # One encode call gives us BOTH the starting latent AND encoder features for fusion.
        inputs_px = inputs.to(device)
        with torch.no_grad():
            posterior, enc_feats = vae.model.encode(inputs_px)
            z_raw = posterior.sample()   # (N, C_lat, H_lat, W_lat)

        # Normalize latent to match the distribution the DiT was trained on
        inputs_y = (z_raw - latent_mean) / latent_std * latent_multiplier
        z        = inputs_y.clone()

        # Setup classifier-free guidance:
        if using_cfg:
            z     = torch.cat([z, z], 0)
            y_cfg = torch.cat([inputs_y, inputs_y], 0)
            model_kwargs = dict(y=y_cfg, inputs=inputs_y, cfg_scale=cfg_scale,
                                cfg_interval=True, cfg_interval_start=cfg_interval_start,
                                # label=torch.cat([label, label], 0)
                                )
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=inputs_y, inputs=inputs_y)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        # Denormalize latent back to VAE space
        samples = (samples * latent_std) / latent_multiplier + latent_mean
        z_in = (inputs_y * latent_std) / latent_multiplier + latent_mean

        #print(samples.max(), samples.min(), z_in.max(), z_in.min(), samples.shape, z_in.shape, inputs_px.shape)
        # Decode with and without fusion - save both results
        # Decode with enc_feats from the same encode call — perfectly aligned,
        # no interpolation needed.
        samples_np_fusion = vae.decode_to_images(samples, feats=enc_feats)   # (N, H_pix, W_pix, 3)
        # Decode without fusion
        samples_np_nofusion = vae.decode_to_images(samples, feats=None)     # (N, H_pix, W_pix, 3)
        samples_input_np = vae.decode_to_images(z_in, feats=enc_feats)     # (N, H_pix, W_pix, 3)

        gts_pixel    = (gts    + 1.0) / 2.0          # (N, 3, H_pad_pix, W_pad_pix)
        inputs_pixel = (inputs + 1.0) / 2.0

        # Save samples – resize back to original size, keep strictly sequential indices
        batch_size = samples_np_fusion.shape[0]
        for i in range(batch_size):
            orig_pix_h, orig_pix_w = original_pixel_sizes[i]
            
            index = i * accelerator.num_processes + accelerator.process_index + total
            
            # Resize decoded samples back to original pixel size
            def resize_np_to_original(img_np, target_h, target_w):
                """Resize numpy array (H, W, 3) to (target_h, target_w, 3)"""
                pil_img = Image.fromarray(img_np)
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                return np.array(pil_img)
            
            # Resize and save with fusion result
            resized_fusion = resize_np_to_original(samples_np_fusion[i], orig_pix_h, orig_pix_w)
            Image.fromarray(resized_fusion).save(
                f"{sample_folder_dir}/{index:06d}.png")
            
            # Resize and save without fusion result
            resized_nofusion = resize_np_to_original(samples_np_nofusion[i], orig_pix_h, orig_pix_w)
            Image.fromarray(resized_nofusion).save(
                f"{sample_folder_dir}/{index:06d}_nofusion.png")
            
            # Resize and save input reconstruction
            resized_input = resize_np_to_original(samples_input_np[i], orig_pix_h, orig_pix_w)
            Image.fromarray(resized_input).save(
                f"{sample_folder_dir}/{index:06d}_vae_input.png")
            
            # For GT and input, resize from 720x720 back to original
            gt_720 = (gts_pixel[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            in_720 = (inputs_pixel[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            gt_pil = Image.fromarray(gt_720).resize((orig_pix_w, orig_pix_h), Image.LANCZOS)
            in_pil = Image.fromarray(in_720).resize((orig_pix_w, orig_pix_h), Image.LANCZOS)
            
            gt_np = np.array(gt_pil)
            in_np = np.array(in_pil)
            
            Image.fromarray(gt_np).save(f"{sample_folder_dir}/{index:06d}_gt.png")
            Image.fromarray(in_np).save(f"{sample_folder_dir}/{index:06d}_input.png")

        # Increment by actual samples processed across all processes this step
        total += batch_size * accelerator.num_processes
    accelerator.wait_for_everyone()
    # Calculate metrics for both fusion and no fusion versions
    # Initialize LPIPS model once for reuse
    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net='vgg').eval()
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
    avg_psnr, avg_ssim, avg_lpips = calculate_metrics_for_folder(sample_folder_dir, accelerator, no_fusion=False, lpips_model=lpips_model)
    avg_psnr_nofusion, avg_ssim_nofusion, avg_lpips_nofusion = calculate_metrics_for_folder(sample_folder_dir, accelerator, no_fusion=True, lpips_model=lpips_model)
    return avg_psnr, avg_ssim, avg_lpips, avg_psnr_nofusion, avg_ssim_nofusion, avg_lpips_nofusion

def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_b_ldmvae_f16d16.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    accelerator = Accelerator()
    original_config = load_config(args.config)

    # get ckpt_dir
    assert 'ckpt_path' in original_config, "ckpt_path must be specified in config"
    ckpt_dir = original_config['ckpt_path']

    if 'downsample_ratio' in original_config['vae']:
        latent_size = original_config['data']['image_size'] // original_config['vae']['downsample_ratio']
    else:
        latent_size = original_config['data']['image_size'] // 16

    if accelerator.process_index == 0:
        print_with_prefix('Loading LightningDiT Model...')
        
    model = LightningDiT_models[original_config['model']['model_type']](
        input_size=latent_size,
        num_classes=original_config['data']['num_classes'],
        use_qknorm=original_config['model']['use_qknorm'],
        use_swiglu=original_config['model']['use_swiglu'] if 'use_swiglu' in original_config['model'] else False,
        use_rope=original_config['model']['use_rope'] if 'use_rope' in original_config['model'] else False,
        use_rmsnorm=original_config['model']['use_rmsnorm'] if 'use_rmsnorm' in original_config['model'] else False,
        wo_shift=original_config['model']['wo_shift'] if 'wo_shift' in original_config['model'] else False,
        in_channels=original_config['model']['in_chans'] if 'in_chans' in original_config['model'] else 4,
        learn_sigma=original_config['model']['learn_sigma'] if 'learn_sigma' in original_config['model'] else False,
    )

    checkpoint = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    incompatible_keys = model.load_state_dict(checkpoint, strict=False)
    if accelerator.process_index == 0:
        missing_keys = incompatible_keys.missing_keys
        unexpected_keys = incompatible_keys.unexpected_keys
        if missing_keys:
            print_with_prefix(f"Missing keys ({len(missing_keys)}):")
            for key in missing_keys:
                print_with_prefix(f"  - {key}")
        else:
            print_with_prefix("No missing keys.")
        if unexpected_keys:
            print_with_prefix(f"Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys:
                print_with_prefix(f"  - {key}")
        else:
            print_with_prefix("No unexpected keys.")
    model.eval()
    model.to(accelerator.device)

    # Enable training-free multi-resolution inference:
    # timm's PatchEmbed asserts H==img_size[0] before the conv projection.
    # Setting img_size=None disables that check while keeping all weights intact.
    for embedder_name in ('x_embedder', 'y_embedder', 'y_embedder_weight'):
        embedder = getattr(model, embedder_name, None)
        if embedder is not None:
            embedder.img_size = None
    if accelerator.process_index == 0:
        print_with_prefix('Loading VAE Model...')
        
    vae = VA_VAE(f'tokenizer/configs/{original_config["vae"]["model_name"]}.yaml')
    fusion_ckpt = original_config['vae'].get('fusion_ckpt_path', None)
    print(fusion_ckpt)
    if fusion_ckpt:
        vae.load_fusion_layer(fusion_ckpt)
    
    base_exp_name = original_config['train']['exp_name']
    results_table = []
    for data_path in DATASETS_LIST:
       
        current_config = deepcopy(original_config)
        dataset_name = os.path.basename(data_path.rstrip('/'))
        
        current_config['data']['data_path'] = data_path
        current_config['train']['exp_name'] = os.path.join(base_exp_name, dataset_name)
        
        if accelerator.process_index == 0:
            print_with_prefix(f"---- Processing Dataset: {dataset_name} ----")
        
       
        try:
            psnr, ssim, lpips_score, psnr_nofusion, ssim_nofusion, lpips_nofusion = do_sample(
                current_config, 
                accelerator, 
                ckpt_path=ckpt_dir, 
                model=model, 
                vae=vae, 
                demo_sample_mode=args.demo
            )
            if accelerator.is_main_process:
                    results_table.append([dataset_name, f"{psnr:.2f}", f"{ssim:.4f}", f"{lpips_score:.4f}", f"{psnr_nofusion:.2f}", f"{ssim_nofusion:.4f}", f"{lpips_nofusion:.4f}"])
                    print("\n" + "-"*50)
                    print(f"Metrics for Dataset: {dataset_name}")
                    print(f"  > Fusion:    PSNR = {psnr:.2f}, SSIM = {ssim:.4f}")
                    print(f"  > No Fusion: PSNR = {psnr_nofusion:.2f}, SSIM = {ssim_nofusion:.4f}")
                    print("-"*50 + "\n")
                    
        except Exception as e:
            print_with_prefix(f"Error in {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()
    
    if accelerator.process_index == 0:
        print("\n" + "="*120)
        print(f"{'Dataset Name':<30} | {'PSNR (F)':<12} | {'SSIM (F)':<12} | {'LPIPS (F)':<12} | {'PSNR (NF)':<12} | {'SSIM (NF)':<12} | {'LPIPS (NF)':<12}")
        print("-" * 120)
        for row in results_table:
            print(f"{row[0]:<30} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12} | {row[4]:<12} | {row[5]:<12} | {row[6]:<12}")
        print("="*120 + "\n")
        print_with_prefix("All datasets processed and evaluated.")
