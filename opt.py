import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'colmap_exr', 'colmap_real_exr', 'myblender', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--depth_loss_w', type=float, default=0,
                        help='''weight of depth loss (see losses.py),
                        to reduce floaters near the camera
                        ''')
    parser.add_argument('--loss_func', type=str, default='raw',
                        choices=['raw', 'log', 'tanh'],
                        help='select loss function for HDR training')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')

    # NOTE: if use lego-like dataset, should not use random bg, because it has black/white bg
    # for other full-screen image dataset, it recommands to enable random_bg
    # to avoid holes when the color equals to the fix bg
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--val_batch_size', type=int, default=2**20,
                        help='number of rays in a batch when validation')
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # GUI
    parser.add_argument('--low_resolution', type=float, default=1.0,
                        help='render size divide by this')

    # Insertor
    parser.add_argument('--max_pc_pts_num', type=int, default=int(1e6),
                        help='max pc pts num the insertor generate')
    parser.add_argument('--no_global_SH', action='store_true', default=False,
                        help='do not train global SH')
    
    # HDR
    '''
    if use LDR imgs dataset: all set to false
    if use HDR imgs dataset: 
        if use global SH, but LDR output, set train_SH_HDR_mapping true
        ...               but HDR output, set train_SH_HDR_mapping false
        set gen_probe_HDR_mapping false
        set render_HDR_mapping true
    '''
    parser.add_argument('--train_SH_HDR_mapping', action='store_true', default=False,
                        help='use HDR mapping when inverse rendering')
    parser.add_argument('--gen_probe_HDR_mapping', action='store_true', default=False,
                        help='use HDR mapping when gen probe')
    parser.add_argument('--render_HDR_mapping', action='store_true', default=False,
                        help='use HDR mapping when render mesh')

    parser.add_argument('--use_EXR', action='store_true', default=False,
                        help='use exr HDR images to train/render')

    return parser.parse_args()