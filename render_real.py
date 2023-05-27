import sys, os, imageio, lpips

root = '/data/best_1023/mvsnerf_head_eca_drop'
os.chdir(root)
sys.path.append(root)

from opt import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# models
from models import *
from renderer import *
from data.ray_utils import get_rays

from tqdm import tqdm

from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers

from data.ray_utils import ray_marcher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# def decode_batch(batch):
#     rays = batch['rays']  # (B, 8)
#     rgbs = batch['rgbs']  # (B, 3)
#     return rays, rgbs


def unpreprocess(data, shape=(1, 1, 3, 1, 1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std

def decode_batch(batch, idx=list(torch.arange(4))):

    data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
    pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                'c2ws': data_mvs['c2ws'].squeeze(), 'near_fars': data_mvs['near_fars'].squeeze()}

    return data_mvs, pose_ref

loss_fn_vgg = lpips.LPIPS(net='vgg')
mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)
max_psnr = 0
ckpt_path = "best_128"
ckpts = os.listdir(f"/data/best_1023/mvsnerf_head_eca_drop/{ckpt_path}")
ckpt_remove = []
for i in ckpt_remove:
    ckpts.remove(i)
# ckpts = ['59999.tar']
print(ckpts)
for ckpt in ckpts:
    dir_ckpt = ckpt.split('.')[0]
    
    save_dir = f'/data/best_1023/mvsnerf_head_eca_drop/results/{ckpt_path}/nerf_real_{dir_ckpt}'
    os.makedirs(save_dir, exist_ok=True)
    f_txt = open(save_dir + '/' + 'out.txt', "w")
    f_txt.write(f'test_{save_dir} \n')
    psnr_all, ssim_all, LPIPS_vgg_all = [], [], []
    depth_acc = {}
    eval_metric = [0.1, 0.05, 0.01]
    depth_acc[f'abs_err'], depth_acc[f'acc_l_{eval_metric[0]}'], depth_acc[f'acc_l_{eval_metric[1]}'], depth_acc[
        f'acc_l_{eval_metric[2]}'] = {}, {}, {}, {}

    cmd = f'--datadir /dataset/mvsnerf/realestate_test  \
        --dataset_name real  \
        --net_type v0 --ckpt ./{ckpt_path}/{ckpt}'  # 这里需要修改
    args = config_parser(cmd.split())
    dataset_val = dataset_dict[args.dataset_name](args, 'val')


    psnr, ssim, LPIPS_vgg = [], [], []


    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim = 8 + 12  # 修改

        # create models

    render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True,
                                                                                dir_embedder=False,
                                                                                pts_embedder=True)
    filter_keys(render_kwargs_train)

    MVSNet = render_kwargs_train['network_mvs']
    render_kwargs_train.pop('network_mvs')

    # print(MVSNet)
    datadir = args.datadir
    datatype = 'val'
    pad = 24
    args.chunk = 5120
    args.netchunk = 5120
    print('============> rendering dataset <===================')

    save_as_image = True
    MVSNet.train()
    MVSNet = MVSNet.cuda()
        
    val_dataset = DataLoader(dataset_val,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True)

    with torch.no_grad():

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        for i, batch in enumerate(val_dataset):
            
            data_mvs, pose_ref = decode_batch(batch)
            # print(batch['near_fars'])
            imges, proj_mats = data_mvs['images'], data_mvs['proj_mats']
            near_fars = pose_ref['near_fars']
            depths_h = data_mvs['depths_h']
            H, W = imges.shape[-2:]
            H, W = int(H), int(W)

            world_to_ref = pose_ref['w2cs'][0]
            tgt_to_world, intrinsic = pose_ref['c2ws'][-1], pose_ref['intrinsics'][-1]
            volume_feature, img_feat, depth_values = MVSNet(imges[:, :3], proj_mats[:, :3], near_fars[0],
                                                            pad=args.pad)
            # volume_feature = torch.zeros((1, 8, 128, 176, 208)).float()
            imgs = unpreprocess(imges)

            rgb_rays, depth_rays_preds = [], []
            for chunk_idx in range(H * W // args.chunk + int(H * W % args.chunk > 0)):
                rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters = \
                    build_rays_test(H, W, tgt_to_world, world_to_ref, intrinsic, near_fars, near_fars[-1],
                                    args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx)
                rgb, disp, acc, depth_pred, density_ray, ret = rendering(args, pose_ref, rays_pts, rays_NDC,
                                                                        depth_candidates, rays_o, rays_dir,
                                                                        volume_feature, imgs[:, :-1], img_feat=None,
                                                                        **render_kwargs_train)
                rgb, depth_pred = torch.clamp(rgb.cpu(), 0, 1.0).numpy(), depth_pred.cpu().numpy()
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)
            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
            img1 = imgs[0, 0].permute(1, 2, 0).cpu().numpy()
            img2 = imgs[0, 1].permute(1, 2, 0).cpu().numpy()
            img3 = imgs[0, 2].permute(1, 2, 0).cpu().numpy()
            img = imgs[0, 3].permute(1, 2, 0).cpu().numpy()
            img_vis = np.concatenate((img1*255,img2*255,img3*255,img*255,rgb_rays*255),axis=1)
            if save_as_image:
                print(f'{save_dir}/scan_{i:03d}.png')
                imageio.imwrite(f'{save_dir}/scan_{i:03d}.png', img_vis.astype('uint8'))
            else:
                rgbs.append(img_vis.astype('uint8'))
            # print(rgb_rays.shape)
            # print(img.shape)
            mpsnr = mse2psnr(np.mean((rgb_rays - img) ** 2))
            mssin = structural_similarity(rgb_rays, img, multichannel=True)
            
            psnr.append(mpsnr)
            ssim.append(mssin)
            img_tensor = torch.from_numpy(rgb_rays)[None].permute(0, 3, 1,
                                        2).float() * 2 - 1.0  # image should be RGB, IMPORTANT: normalized to [-1,1]
            img_gt_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
            lpips = loss_fn_vgg(img_tensor, img_gt_tensor).item()
            LPIPS_vgg.append(lpips)
            print(mpsnr, mssin, lpips)
            f_txt.write(f'=====> psnr {mpsnr} ssim: {mssin} lpips: {lpips} \n')
    print(f'=====> all mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')
    f_txt.write(f'=====> all mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)} \n')
    f_txt.close()
    mean_psnr = np.mean(psnr_all)
    # if max_psnr < mean_psnr:
    #     max_psnr = mean_psnr
    #     max_psnr_skpt = ckpt
    #     print(f'max psnr now {max_psnr}')
# print(f'max spnr ckpt is ' + max_psnr_skpt)  
