import os
import time
import numpy as np
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
from torch.utils.data import DataLoader

from models.restormer_arch import Restormer_PPFN
from models.predictor import DegradationPredictor
import torch.nn.functional as F

from dataset import PairedTrainSetLoader, PairedEvalSetLoader
import argparse
import utils
from utils import network_parameters
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument("--dataset_name", default='M3FD', type=str, help="dataset_name")
parser.add_argument("--train_split", default='train', type=str, help="training split folder")
parser.add_argument("--val_split", default='test', type=str, help="validation/test split folder used during training")
parser.add_argument("--dataset_dir", default='./datasets/M3FD_Detection', type=str, help="train_dataset_dir")
parser.add_argument("--save_dir", default='./exps', type=str, help="Save path of checkpoints")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patch_size", type=int, default=48, help="Training patch size")
parser.add_argument("--acc_step", type=int, default=1, help="Training accelerate step")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--val_epoch", type=int, default=20, help="Validation of epoch")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
parser.add_argument("--num_degradation_types", type=int, default=4, help="number of degradation labels")
parser.add_argument("--paired_fast", action="store_true",
                    help="For paired input/target data, train and validate only one restoration stage")
parser.add_argument("--paired_fast_stage", type=int, default=0, choices=[0, 1, 2],
                    help="Stage index used by --paired_fast")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr_initial': 8e-5, 'lr_min': 1e-6}, type=dict,
                    help="optimizer settings")
parser.add_argument("--lr_initial", default=None, type=float,
                    help="Initial learning rate. Overrides optimizer_settings['lr_initial'] when set")
parser.add_argument("--lr_min", default=None, type=float,
                    help="Minimum learning rate. Overrides optimizer_settings['lr_min'] when set")
parser.add_argument("--predictor_lr", default=None, type=float,
                    help="Learning rate for degradation predictor. Defaults to lr_initial when set")
parser.add_argument("--prompt_lr", default=None, type=float,
                    help="Learning rate for prompt bank. Defaults to lr_initial when set")
parser.add_argument("--pretrain_weights", default='', type=str,
                    help="Load model/predictor/prompt weights but start training from epoch 1")
parser.add_argument("--prompt_only", action="store_true",
                    help="Freeze the restoration backbone and train only prompt adapters, predictor, and prompt bank")
parser.add_argument("--resume", default=False, type=bool, help="use resumed model parameters")
args = parser.parse_args()
if args.resume and args.pretrain_weights:
    raise ValueError("--resume and --pretrain_weights are mutually exclusive")
if not os.path.exists(args.dataset_dir) and os.path.exists('./datasets'):
    print(f"==> Dataset root {args.dataset_dir} not found. Falling back to ./datasets")
    args.dataset_dir = './datasets'

## Set Seeds
seed = args.seed
torch.backends.cudnn.benchmark = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


lr_initial = args.lr_initial if args.lr_initial is not None else args.optimizer_settings['lr_initial']
lr_min = args.lr_min if args.lr_min is not None else args.optimizer_settings['lr_min']
predictor_lr = args.predictor_lr if args.predictor_lr is not None else lr_initial
prompt_lr = args.prompt_lr if args.prompt_lr is not None else lr_initial
OPT = {'BATCH': args.batch_size, 'EPOCHS': args.epochs, 'LR_INITIAL': lr_initial,
       'LR_MIN': lr_min, 'PATCH': args.patch_size,}


def load_state_dict_compatible(module, state_dict):
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if all(k.startswith('module.') for k in state_dict.keys()):
        stripped_state_dict = {k[7:]: v for k, v in state_dict.items()}
        try:
            module.load_state_dict(stripped_state_dict)
            return
        except RuntimeError:
            pass

    prefixed_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    module.load_state_dict(prefixed_state_dict)


def set_prompt_only_trainable(model):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in model.parameters():
        param.requires_grad = False

    prompt_prefixes = ('F_ext_net', 'prompt_scale', 'prompt_shift')
    for name, module in base_model.named_modules():
        if name.startswith(prompt_prefixes):
            for param in module.parameters():
                param.requires_grad = True

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def pad_to_multiple(tensor, multiple=8):
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor

    pad_mode = 'reflect' if height > pad_h and width > pad_w else 'replicate'
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)


## Build Model
print('==> Build the model')
model_restored = Restormer_PPFN(inp_channels=1, out_channels=1)
r_number = network_parameters(model_restored)
type_npl = np.load(os.path.join(os.getcwd(), 'data/type7.npy')).astype(np.float32)
typep = torch.from_numpy(type_npl[:3]).clone().cuda()
# multip is no longer fixed, but we might still use it for initialization or reference if needed
# multi_npl = np.load(os.path.join(os.getcwd(), 'data/multi2.npy')).astype(np.float32)
# multip = torch.from_numpy(multi_npl).clone().cuda()
model_restored.cuda()

# 1. Initialize degradation predictor
num_degradation_types = args.num_degradation_types # 0: clean, 1-3: degraded classes
predictor = DegradationPredictor(in_channels=1, num_classes=num_degradation_types).cuda()

# 2. Define learnable Prompt Bank
prompt_dim = 64  # Prompt dimension expected by Restormer_PPFN.
prompt_bank = nn.Parameter(torch.randn(num_degradation_types, prompt_dim).cuda())

## Training model path direction
mode = 'Restormer-PPFN_' + args.dataset_name + '_train'

model_dir = os.path.join(args.save_dir, mode, 'models')
utils.mkdir(model_dir)

## GPU
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
    predictor = nn.DataParallel(predictor, device_ids=device_ids)

if args.prompt_only:
    trainable_model_params = set_prompt_only_trainable(model_restored)
    print(f"==> Prompt-only tuning enabled; trainable restoration params: {trainable_model_params}")

## Optimizer
start_epoch = 1
acc_step = args.acc_step
lr = float(OPT['LR_INITIAL'])
r_optimizer = optim.Adam([
    {'params': [p for p in model_restored.parameters() if p.requires_grad]},
    {'params': predictor.parameters(), 'lr': predictor_lr},
    {'params': [prompt_bank], 'lr': prompt_lr}
],
    lr=lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(r_optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(r_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

## EMA
ema_model = utils.ModelEMA(model_restored.state_dict(), decay=0.999)

if args.pretrain_weights:
    print('------------------------------------------------------------------')
    print(f"==> Loading pretrained weights: {args.pretrain_weights}")
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu')

    model_state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    load_state_dict_compatible(model_restored, model_state_dict)

    if isinstance(checkpoint, dict) and 'predictor_state_dict' in checkpoint:
        try:
            load_state_dict_compatible(predictor, checkpoint['predictor_state_dict'])
            print("==> Loaded pretrained predictor state dict")
        except Exception as e:
            print(f"==> Failed to load pretrained predictor: {e}")

    if isinstance(checkpoint, dict) and 'prompt_bank' in checkpoint:
        try:
            pretrained_prompt_bank = checkpoint['prompt_bank'].detach().to(prompt_bank.device)
            if pretrained_prompt_bank.shape != prompt_bank.shape:
                raise ValueError(f"prompt_bank shape mismatch: checkpoint {tuple(pretrained_prompt_bank.shape)} vs current {tuple(prompt_bank.shape)}")
            prompt_bank.data.copy_(pretrained_prompt_bank)
            print("==> Loaded pretrained prompt bank")
        except Exception as e:
            print(f"==> Failed to load pretrained prompt bank: {e}")

    ema_model = utils.ModelEMA(model_restored.state_dict(), decay=0.999)
    print("==> Pretrained weights loaded; training will start from epoch 1")
    print('------------------------------------------------------------------')

## Resume (Continue training by a pretrained model)
if args.resume:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest, 'state_dict')
    
    # Load predictor and prompt_bank if available
    checkpoint = torch.load(path_chk_rest)
    if 'predictor_state_dict' in checkpoint:
        try:
            predictor.load_state_dict(checkpoint['predictor_state_dict'])
            print("==> Loaded predictor state dict")
        except Exception as e:
            print(f"==> Failed to load predictor: {e}")
            
    if 'prompt_bank' in checkpoint:
        try:
            prompt_bank.data = checkpoint['prompt_bank'].data
            print("==> Loaded prompt bank")
        except Exception as e:
            print(f"==> Failed to load prompt bank: {e}")
            
    if 'ema_state_dict' in checkpoint:
        try:
            ema_model.ema_state_dict = checkpoint['ema_state_dict']
            print("==> Loaded EMA state dict")
        except Exception as e:
            print(f"==> Failed to load EMA state dict: {e}")

    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(r_optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
loss_char = utils.CharbonnierLoss().cuda()
loss_edge = utils.EdgeLoss().cuda()
CE_loss = nn.CrossEntropyLoss()



## DataLoaders
print('==> Loading datasets')

train_dataset = PairedTrainSetLoader(args.dataset_dir, args.train_split, (OPT['PATCH'], OPT['PATCH']))
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
val_dataset = PairedEvalSetLoader(args.dataset_dir, args.val_split)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
stage_indices = [args.paired_fast_stage] if args.paired_fast else [0, 1, 2]

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Model parameters:   {r_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    Train split:        {args.dataset_dir}/{args.train_split}
    Val/Test split:     {args.dataset_dir}/{args.val_split}
    Paired fast:        {args.paired_fast}
    Stage indices:      {stage_indices}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(args.save_dir, mode, 'log', time.strftime("%Y%m%d_%H%M%S"))
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix='_train')
text_log_path = os.path.join(args.save_dir, mode, 'train_log.csv')
if not os.path.exists(text_log_path):
    with open(text_log_path, 'w') as f:
        f.write('epoch,mode,psnr,ssim,loss,lr,time\n')
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_r_loss = []
    epoch_p_loss = []

    model_restored.train()
    r_optimizer.zero_grad()
    tbar = tqdm(train_loader)
    for i, data in enumerate(tbar):
        target = data[1].cuda()
        input_ = data[0].cuda()
        m = data[2].cuda()

        total_loss = 0.0
        # input_ comes from train/input only; target from train/target is used only below as loss supervision.
        input_s = input_[:, -1]

        # --- Degradation Prediction & Dynamic Prompting ---
        # Predict degradation type from the input image
        pred_logits = predictor(input_s)
        loss_cls = CE_loss(pred_logits, m.long())

        # Generate dynamic prompt
        weights = F.softmax(pred_logits, dim=1)
        dynamic_mprompt = torch.matmul(weights, prompt_bank)

        # Orthogonal Regularization
        gram_matrix = torch.matmul(prompt_bank, prompt_bank.t())
        identity = torch.eye(num_degradation_types).cuda()
        loss_ortho = ((gram_matrix - identity) ** 2).mean()

        for stage_pos, idx in enumerate(stage_indices):
            restored = model_restored([input_s, typep[2 - idx].unsqueeze(0), dynamic_mprompt])
            tar = target

            # Compute loss
            # loss_rec = 1.0 * L1loss(restored, tar)
            loss_rec = loss_char(restored, tar) + 0.05 * loss_edge(restored, tar)
            
            # Total loss = Reconstruction Loss + Classification Loss + Orthogonal Loss
            loss_each = loss_rec + 0.1 * loss_cls + 0.01 * loss_ortho

            if idx < 2 and not args.paired_fast:
                input_s = restored.clamp(0, 1).detach()
            total_loss += loss_each.detach().cpu().item()

            loss = loss_each / acc_step
            loss.backward(retain_graph=(stage_pos < len(stage_indices) - 1))

        # Back propagation
        if (i + 1) % acc_step == 0 or i == len(train_loader) - 1:
            r_optimizer.step()
            r_optimizer.zero_grad()
            # Update EMA
            ema_model.update(model_restored.state_dict())

        epoch_r_loss.append(total_loss)
        tbar.set_description("Epoch: %d Restoration: loss = %f" % (epoch, np.mean(epoch_r_loss)))
    r_optimizer.zero_grad()


    ## Evaluation (Validation)
    if epoch % args.val_epoch == 0:
        # Backup original weights
        print("==> Switching to EMA weights for validation...")
        backup_state_dict = {k: v.cpu().clone() for k, v in model_restored.state_dict().items()}
        model_restored.load_state_dict(ema_model.ema_state_dict)

        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        err_pos_list = []
        tbar = tqdm(val_loader)
        for ii, data_val in enumerate(tbar):
            with torch.no_grad():
                target = data_val[1].cuda()
                input_ = data_val[0].cuda()
                _, _, ori_h, ori_w = target.shape
                input_ = pad_to_multiple(input_, multiple=8)

                # Predict degradation for validation
                pred_logits = predictor(input_)
                weights = F.softmax(pred_logits, dim=1)
                dynamic_mprompt = torch.matmul(weights, prompt_bank)

                restored = input_
                for idx in stage_indices:
                    restored = model_restored([restored, typep[2-idx].unsqueeze(0), dynamic_mprompt])
                    restored = torch.clamp(restored, 0, 1)

                restored = restored[:, :, :ori_h, :ori_w]
                input_vis = input_[:, :, :ori_h, :ori_w]
                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    ssim_val_rgb.append(utils.torchSSIM(restored, target))

                tbar.set_description("psnr = %.4f, ssim = %.4f" % (torch.stack(psnr_val_rgb).mean().item(), torch.stack(ssim_val_rgb).mean().item()))
        img_grid_i = vutils.make_grid(input_vis[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('input img', img_grid_i, global_step=epoch)
        img_grid_o = vutils.make_grid(restored[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('output img', img_grid_o, global_step=epoch)
        img_gt = vutils.make_grid(target[0], normalize=True, scale_each=True, nrow=8)
        writer.add_image('img gt', img_gt, global_step=epoch)

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'predictor_state_dict': predictor.state_dict(),
                        'prompt_bank': prompt_bank,
                        'optimizer': r_optimizer.state_dict(),
                        'ema_state_dict': ema_model.ema_state_dict
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'predictor_state_dict': predictor.state_dict(),
                        'prompt_bank': prompt_bank,
                        'optimizer': r_optimizer.state_dict(),
                        'ema_state_dict': ema_model.ema_state_dict
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
        
        # Restore original weights
        print("==> Restoring original weights for training...")
        model_restored.load_state_dict(backup_state_dict)
        
    scheduler.step()

    epoch_time = time.time() - epoch_start_time
    current_lr = scheduler.get_lr()[0]

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, epoch_time,
                                                                              np.mean(epoch_r_loss), current_lr))
    print("------------------------------------------------------------------")

    log_psnr = best_psnr if epoch % args.val_epoch == 0 else ''
    log_ssim = best_ssim if epoch % args.val_epoch == 0 else ''
    with open(text_log_path, 'a') as f:
        f.write(f"{epoch},train,{log_psnr},{log_ssim},{np.mean(epoch_r_loss):.6f},{current_lr:.8f},{epoch_time:.4f}\n")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'prompt_bank': prompt_bank,
                'optimizer': r_optimizer.state_dict(),
                'ema_state_dict': ema_model.ema_state_dict
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', np.mean(epoch_r_loss), epoch)
    writer.add_scalar('train/lr', current_lr, epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
