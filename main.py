import torch
import numpy as np
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torchvision
from FPADDM.functions import create_img_masks
from FPADDM.models import FPADDMNet, MultiFPAGaussianDiffusion
from FPADDM.trainer import MultiexposureTrainer
from text2live_util.clip_extractor import ClipExtractor
from datasets import MixedNIR2_Dai
from torch.utils.data import DataLoader

def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument("--scope", help='choose training scope.', default='forest')
    parser.add_argument("--mode", help='choose mode: train, deartifacts, clip_roi, harmonization, style_transfer, roi')
    # relevant if mode==hamonization/style_transfer
    parser.add_argument("--input_image", help='image name or image path, which is ready to deartifacts', default='0001B_100ms.png')
    parser.add_argument("--exp_t", help='exporesure time of single image to deartifacts', default=100, type=int)
    # parser.add_argument("--start_t_harm", help='starting T at last scale for harmonization', default=5, type=int)
    # parser.add_argument("--start_t_style", help='starting T at last scale for style transfer', default=15, type=int)
    # relevant if mode==deartifacts
    # parser.add_argument("--noise_mask", help='approximate artifacts mask.', default='100ms_mask.png')
    # relevant if mode==clip_{content/style_gen/style_trans/roi}
    # parser.add_argument("--clip_text", help='enter CLIP text.', default='Fire in the Forest')
    # # relevant if mode==clip_content
    # parser.add_argument("--fill_factor", help='Dictates relative amount of pixels to be changed. Should be between 0 and 1.', type=float)
    # parser.add_argument("--strength", help='Dictates the relative strength of CLIPs gradients. Should be between 0 and 1.',  type=float)
    # parser.add_argument("--roi_n_tar", help='Defines the number of target ROIs in the new image.', default=1, type=int)
    # Dataset
    parser.add_argument("--dataset_folder", help='choose datasets root folder.', default='./datasets/')
    parser.add_argument("--dataset_name", help='choose specific dataset.', default='micexxx')
    parser.add_argument("--FPA_mask_source", help='the source .raw file to generate artifacts masks. (should be 8-bit)', default='./0-100ms.raw')
    # parser.add_argument("--image_name", help='choose image name.', default='forest.jpeg')
    parser.add_argument("--results_folder", help='choose results folder.', default='./results/')
    # Net
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=160, type=int)
    # diffusion params
    # parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1.411, type=float)
    # training params
    parser.add_argument("--timesteps", help='total diffusion timesteps. note that the max trained exporesure time is args.timesteps * 2ms', default=50, type=int)
    parser.add_argument("--train_batch_size", help='batch size during training.', default=8, type=int)
    parser.add_argument("--total_epoch", help='total epochs for training', default=200, type=int)
    parser.add_argument("--start_save_epoch", help='the first epoch for saving', default=1, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    # parser.add_argument("--train_num_steps", help='total training steps.', default=120001, type=int)
    parser.add_argument("--save_and_sample_every", help='n. epochs for checkpointing model.', default=1, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-3, type=float)
    parser.add_argument("--sched_k_epoch", nargs="+", help='lr scheduler steps.',
                        default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_epoch", help='load specific milestone. exactly refer to the weights of specific epoch', default=0, type=int)
    # sampling params
    parser.add_argument("--deart_batch_size", help='batch size during deartifacts.', default=16, type=int)
    # parser.add_argument("--scale_mul", help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    # parser.add_argument("--sample_t_list", nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)
    # device num
    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)

    # DEV. params - do not modify
    # parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega", help='sigma=omega*max_sigma.', default=0, type=float)
    parser.add_argument("--loss_factor", help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    # scale_mul = (args.scale_mul[0], args.scale_mul[1])
    sched_epochs = [val for val in args.sched_k_epoch]
    results_folder = args.results_folder + '/' + args.dataset_name

    # set to true to save all intermediate diffusion timestep results
    save_interm = False

    masks = create_img_masks(args.FPA_mask_source, args.timesteps,
                            create=True,
                        #   auto_scale=50000, # limit max number of pixels in image
                                )
    assert len(masks) == args.timesteps, "Should len(masks) == args.timesteps."
    
    model = FPADDMNet(
        dim=args.dim,
        multiexposure=True,
        device=device
    )
    model.to(device)

    n_exposures = args.timesteps
    fpa_diffusion = MultiFPAGaussianDiffusion(
        denoise_fn=model,
        save_interm=save_interm,
        results_folder=results_folder, # for debug
        n_exposures=n_exposures,
        # scale_factor=scale_factor,
        # image_sizes=sizes,
        # scale_mul=scale_mul,
        channels=3,
        timesteps=args.timesteps,
        train_full_t=True,
        # scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l1',
        betas=None,
        device=device,
        # reblurring=True,
        # sample_limited_t=args.sample_limited_t,
        omega=args.omega,
        masks=masks
    ).to(device)

    # if args.sample_t_list is None:
    #     sample_t_list = fpa_diffusion.num_timesteps_ideal[1:]
    # else:
    #     sample_t_list = args.sample_t_list

    # prepare dataloader
    
    dataset_train = MixedNIR2_Dai.FPADDMDataset(root_dir=args.dataset_folder, split='train')
    dataset_val = MixedNIR2_Dai.FPADDMDataset(root_dir=args.dataset_folder, split='val')
    train_loader = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.deart_batch_size, shuffle=False)
    dataloader_list = [train_loader, val_loader]

    ExposureTrainer = MultiexposureTrainer(
            fpa_diffusion,
            folder=args.dataset_folder,
            dataloader_list=dataloader_list,
            n_exposures=n_exposures,
            # scale_factor=scale_factor,
            # image_sizes=sizes,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            total_epoch=args.total_epoch,  # total training steps
            gradient_accumulate_every=args.grad_accumulate,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=False,  # turn on mixed precision training with apex
            start_save_epoch=args.start_save_epoch,
            avg_window=args.avg_window,
            sched_epochs=sched_epochs,
            results_folder=results_folder,
            device=device,
            raw_path=args.FPA_mask_source,

        )

    if args.load_milestone > 0:
        ExposureTrainer.load(milestone=args.load_milestone)
    if args.mode == 'train':
        ExposureTrainer.train()
        # Sample after training is complete
        # ExposureTrainer.iterative_denoise(scale_mul=(1, 1),    # H,W
        #                            custom_sample=True,
        #                            image_name=args.image_name,
        #                            batch_size=args.sample_batch_size,
        #                            custom_t_list=sample_t_list
        #                            )
    elif args.mode == 'deartifacts':
        ExposureTrainer.train(dataloader_list=[val_loader], total_epoch=1)
    
    # elif args.mode == 'sample':

    #     # # Sample
    #     ExposureTrainer.sample_scales(scale_mul=scale_mul,    # H,W
    #                                custom_sample=True,
    #                                image_name=args.image_name,
    #                                batch_size=args.sample_batch_size,
    #                                custom_t_list=sample_t_list,
    #                                save_unbatched=True
    #                                )

    # elif args.mode == 'clip_content':
    #     # CLIP
    #     text_input = args.clip_text
    #     clip_cfg = {"clip_model_name": "ViT-B/32",
    #                 "clip_affine_transform_fill": True,
    #                 "n_aug": 16}
    #     t2l_clip_extractor = ClipExtractor(clip_cfg)
    #     clip_custom_t_list = sample_t_list

    #     # number of gradient steps per diffusion step for each scale
    #     guidance_sub_iters = [0]
    #     for i in range(n_exposures-1):
    #         guidance_sub_iters.append(1)

    #     assert args.strength is not None and 0 <= args.strength <= 1, f"Strength value should be between 0 & 1. Got: {args.strength} "
    #     assert args.fill_factor is not None and 0 <= args.fill_factor <= 1, f"fill_factor value should be between 0 & 1. Got: {args.fill_factor} "
    #     strength = args.strength
    #     quantile = 1. - args.fill_factor


    #     llambda = 0.2
    #     stop_guidance = 3  # at the last scale, disable the guidance in the last x steps in order to avoid artifacts from CLIP
    #     ExposureTrainer.ema_model.reblurring = False
    #     ExposureTrainer.clip_sampling(clip_model=t2l_clip_extractor,
    #                                text_input=text_input,
    #                                strength=strength,
    #                                sample_batch_size=args.sample_batch_size,
    #                                custom_t_list=clip_custom_t_list,
    #                                quantile=quantile,
    #                                guidance_sub_iters=guidance_sub_iters,
    #                                stop_guidance=stop_guidance,
    #                                save_unbatched=True,
    #                                scale_mul=scale_mul,
    #                                llambda=llambda
    #                                )

    # elif args.mode == 'clip_style_trans' or args.mode == 'clip_style_gen':
    #     # CLIP
    #     text_input = args.clip_text + ' Style'
    #     clip_cfg = {"clip_model_name": "ViT-B/32",
    #                 "clip_affine_transform_fill": True,
    #                 "n_aug": 16}
    #     t2l_clip_extractor = ClipExtractor(clip_cfg)
    #     clip_custom_t_list = sample_t_list

    #     guidance_sub_iters = []
    #     for i in range(n_exposures-1):
    #         guidance_sub_iters.append(0)
    #     guidance_sub_iters.append(1)

    #     strength = 0.3
    #     quantile = 0.0 # allow to change the whole image
    #     llambda = 0.05
    #     stop_guidance = 3  # at the last scale, disable the guidance in the last x steps in order to avoid artifacts from CLIP
    #     if args.mode == 'clip_style_gen':
    #         start_noise = True
    #     else:  # mode == 'clip_style_trans':
    #         start_noise = False  # set false to start from original image at last scale
    #     image_name = args.image_name.rsplit( ".", 1 )[ 0 ] + '.png'
    #     ExposureTrainer.ema_model.reblurring = False
    #     ExposureTrainer.clip_sampling(clip_model=t2l_clip_extractor,
    #                                text_input=text_input,
    #                                strength=strength,
    #                                sample_batch_size=args.sample_batch_size,
    #                                custom_t_list=clip_custom_t_list,
    #                                quantile=quantile,
    #                                guidance_sub_iters=guidance_sub_iters,
    #                                stop_guidance=stop_guidance,
    #                                save_unbatched=True,
    #                                scale_mul=scale_mul,
    #                                llambda=llambda,
    #                                start_noise=start_noise,
    #                                image_name=image_name,
    #                                )

    # elif args.mode == 'clip_roi':
    #     # CLIP_ROI
    #     text_input = args.clip_text
    #     clip_cfg = {"clip_model_name": "ViT-B/32",
    #                 "clip_affine_transform_fill": True,
    #                 "n_aug": 16}
    #     t2l_clip_extractor = ClipExtractor(clip_cfg)
    #     strength = 0.1
    #     num_clip_iters = 100
    #     num_denoising_steps = 3
    #     # select from the finest scale
    #     dataset_folder = os.path.join(args.dataset_folder, f'scale_{n_exposures - 1}/')
    #     image_name = args.image_name.rsplit(".", 1)[0] + '.png'
    #     import cv2
    #     image_to_select = cv2.imread(dataset_folder+image_name)
    #     roi = cv2.selectROI(image_to_select)
    #     lefttop_x = 0
    #     lefttop_y = 0
    #     roi_width = 640
    #     roi_height = 512
    #     roi = (lefttop_x, lefttop_y, roi_width, roi_height)
    #     roi_perm = [1, 0, 3, 2]
    #     roi = [roi[i] for i in roi_perm]
    #     ExposureTrainer.ema_model.reblurring = False
    #     ExposureTrainer.clip_roi_sampling(clip_model=t2l_clip_extractor,
    #                                    text_input=text_input,
    #                                    strength=strength,
    #                                    sample_batch_size=args.sample_batch_size,
    #                                    num_clip_iters=num_clip_iters,
    #                                    num_denoising_steps=num_denoising_steps,
    #                                    clip_roi_bb=roi, #[90,75,50,50],
    #                                    save_unbatched=True,
    #                                    )

    # elif args.mode == 'roi':

    #     import cv2
    #     image_path = os.path.join(args.dataset_folder, f'scale_{n_exposures - 1}', args.image_name.rsplit(".", 1)[0] + '.png')
    #     image_to_select = cv2.imread(image_path)
    #     roi = cv2.selectROI(image_to_select)
    #     lefttop_x = 0
    #     lefttop_y = 0
    #     roi_width = 640
    #     roi_height = 512
    #     roi = (lefttop_x, lefttop_y, roi_width, roi_height)
    #     image_to_select = cv2.cvtColor(image_to_select, cv2.COLOR_BGR2RGB)
    #     roi_perm = [1, 0, 3, 2]
    #     target_roi = [roi[i] for i in roi_perm]
    #     tar_y, tar_x, tar_h, tar_w = target_roi
    #     roi_bb_list = []
    #     n_targets = args.roi_n_tar  # number of target patches
    #     target_h = int(image_to_select.shape[0] * scale_mul[0])
    #     target_w = int(image_to_select.shape[1] * scale_mul[1])
    #     empty_image = np.ones((target_h, target_w, 3))
    #     target_patch_tensor = torchvision.transforms.ToTensor()(
    #         image_to_select[tar_y:tar_y + tar_h, tar_x:tar_x + tar_w, :])

    #     for i in range(n_targets):
    #         roi = cv2.selectROI(empty_image)
    #         roi_reordered = [roi[i] for i in roi_perm]
    #         roi_bb_list.append(roi_reordered)
    #         y, x, h, w = roi_reordered
    #         target_patch_tensor_resize = torch.nn.functional.interpolate(target_patch_tensor[None, :, :, :],
    #                                                                      size=(h, w))
    #         empty_image[y:y + h, x:x + w, :] = target_patch_tensor_resize[0].permute(1, 2, 0).numpy()

    #     empty_image = torchvision.transforms.ToTensor()(empty_image)
    #     torchvision.utils.save_image(empty_image, os.path.join(args.results_folder, args.scope, f'roi_patches.png'))

    #     ExposureTrainer.roi_guided_sampling(custom_t_list=sample_t_list,
    #                                      target_roi=target_roi,
    #                                      roi_bb_list=roi_bb_list,
    #                                      save_unbatched=True,
    #                                      batch_size=args.sample_batch_size,
    #                                      scale_mul=scale_mul)

    # elif args.mode == 'style_transfer' or args.mode == 'harmonization':

    #     i2i_folder = os.path.join(args.dataset_folder, 'i2i')
    #     if args.mode == 'style_transfer':
    #         # start diffusion from last scale
    #         start_s = n_exposures - 1
    #         # start diffusion from t - increase for stronger prior on the original image
    #         start_t = args.start_t_style
    #         use_hist = True
    #     else:
    #         # start diffusion from last scale
    #         start_s = n_exposures - 1
    #         # start diffusion from t - increase for stronger prior on the original image
    #         start_t = args.start_t_harm
    #         use_hist = False
    #     custom_t = []
    #     for i in range(n_exposures-1):
    #         custom_t.append(0)
    #     custom_t.append(start_t)
    #     # use the histogram of the original image for histogram matching
    #     hist_ref_path = f'{args.dataset_folder}scale_{start_s}/'

    #     ExposureTrainer.ema_model.reblurring = True
    #     ExposureTrainer.image2image(input_folder=i2i_folder, input_file=args.input_image, mask=args.harm_mask, hist_ref_path=hist_ref_path,
    #                              batch_size=args.sample_batch_size,
    #                              image_name=args.image_name, start_s=start_s, custom_t=custom_t, scale_mul=(1, 1),
    #                              device=device, use_hist=use_hist, save_unbatched=True, auto_scale=50000, mode=args.mode)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
    quit()
