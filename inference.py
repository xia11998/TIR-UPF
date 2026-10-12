import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import InferSetLoader
from models.predictor import DegradationPredictor
from models.restormer_arch import TIR_UPF


def build_parser():
    parser = argparse.ArgumentParser(description='Hyper-parameters')
    parser.add_argument("--dataset_name", default='Iray_select', type=str, help="dataset_name")
    parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
    parser.add_argument('--output', default='./results', type=str, help='Directory for results')
    parser.add_argument('--cal_iqa', action='store_true', help='Calculating nr-iqa metrics')
    parser.add_argument('--weights', type=str, required=True, help='Path to a checkpoint supplied by the user')
    parser.add_argument('--prompt_mode', default='auto', choices=['auto', 'checkpoint', 'fixed'])
    parser.add_argument('--deg_scenario', default='composited', choices=['single', 'composited'])
    parser.add_argument('--type_order', nargs='*', default=None)
    parser.add_argument('--num_forwards', type=int, default=None)
    return parser


def save_img(filepath, img):
    img.save(filepath)


def default_ablation_config():
    return {
        'use_predictor': True,
        'use_dynamic_prompt': True,
        'use_prompt_modulation': True,
        'use_classification_loss': True,
        'use_orthogonal_loss': True,
        'use_edge_loss': True,
        'global_residual': True,
        'num_stages': 3
    }


def load_state_dict_flexible(module, state_dict):
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    module_state = module.state_dict()
    module_has_prefix = next(iter(module_state)).startswith('module.')
    state_has_prefix = next(iter(state_dict)).startswith('module.')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_has_prefix and not module_has_prefix:
            name = k[7:]
        elif not state_has_prefix and module_has_prefix:
            name = f'module.{k}'
        else:
            name = k
        new_state_dict[name] = v
    module.load_state_dict(new_state_dict)


def tensor_from_checkpoint(checkpoint, key, device):
    tensor = checkpoint.get(key)
    if tensor is None:
        return None
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data
    return tensor.to(device)


def get_stage_prompt_indices(num_stages):
    base = [2, 1, 0]
    if num_stages <= len(base):
        return base[:num_stages]
    return base + [base[-1]] * (num_stages - len(base))


def expand_type_order(order, num_forwards):
    if num_forwards <= len(order):
        return order[:num_forwards]
    return order + [order[-1]] * (num_forwards - len(order))


def build_checkpoint_prompt(input_tensor, checkpoint, predictor, device):
    prompt_bank = tensor_from_checkpoint(checkpoint, 'prompt_bank', device)
    if prompt_bank is not None and predictor is not None:
        pred_logits = predictor(input_tensor)
        weights = F.softmax(pred_logits, dim=1)
        return torch.matmul(weights, prompt_bank)

    static_prompt = tensor_from_checkpoint(checkpoint, 'static_prompt', device)
    if static_prompt is not None:
        return static_prompt.expand(input_tensor.shape[0], -1)

    if prompt_bank is not None:
        uniform_weights = torch.full(
            (input_tensor.shape[0], prompt_bank.shape[0]),
            1.0 / prompt_bank.shape[0],
            device=device,
            dtype=prompt_bank.dtype
        )
        return torch.matmul(uniform_weights, prompt_bank)

    return None


def main():
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location=device)
    config = checkpoint.get('ablation_config', default_ablation_config())
    if not config.get('use_prompt_modulation', True):
        raise ValueError('This backbone requires prompt modulation; use a compatible checkpoint.')

    model = TIR_UPF(
        inp_channels=1,
        out_channels=1,
        global_residual=config.get('global_residual', True)
    ).to(device)
    load_state_dict_flexible(model, checkpoint["state_dict"])
    model.eval()

    predictor = None
    if config.get('use_predictor', True) and 'predictor_state_dict' in checkpoint:
        predictor = DegradationPredictor(in_channels=1, num_classes=4).to(device)
        load_state_dict_flexible(predictor, checkpoint['predictor_state_dict'])
        predictor.eval()

    checkpoint_prompt_available = (
        ('predictor_state_dict' in checkpoint and 'prompt_bank' in checkpoint) or
        'static_prompt' in checkpoint or
        'prompt_bank' in checkpoint
    )
    if args.prompt_mode == 'auto':
        prompt_mode = 'checkpoint' if checkpoint_prompt_available else 'fixed'
    else:
        prompt_mode = args.prompt_mode

    if args.cal_iqa:
        import pyiqa
        niqe = pyiqa.create_metric("niqe", device=device)
        nima = pyiqa.create_metric("nima", device=device)
        musiq = pyiqa.create_metric("musiq", device=device)
        score_niqe = 0.
        score_nima = 0.
        score_musiq = 0.

    out_dir = args.output
    dataset = InferSetLoader(args.dataset_dir, args.dataset_name)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    type_npl = np.load(os.path.join(os.getcwd(), 'data/type7.npy')).astype(np.float32)
    typep = torch.from_numpy(type_npl[:3]).clone().to(device)
    multi_npl = np.load(os.path.join(os.getcwd(), 'data/multi2.npy')).astype(np.float32)
    multip = torch.from_numpy(multi_npl).clone().to(device)
    type_dic = {"contrast": 0, "blur": 1, "noise": 2}
    sce_dic = {"single": 0, "composited": 1}

    if args.type_order is None:
        type_order = ['noise', 'blur', 'contrast'] if args.deg_scenario == 'composited' else ['contrast']
    else:
        type_order = args.type_order

    if prompt_mode == 'checkpoint':
        num_forwards = args.num_forwards or max(1, int(config.get('num_stages', 3)))
        stage_prompt_indices = get_stage_prompt_indices(num_forwards)
    else:
        num_forwards = args.num_forwards or len(type_order)
        type_order = expand_type_order(type_order, num_forwards)

    tbar = tqdm(loader)
    for _, data_val in enumerate(tbar):
        input_ = data_val[0].to(device)
        file_name = data_val[2][0]
        h, w = data_val[1]

        with torch.no_grad():
            restored = input_
            if prompt_mode == 'checkpoint':
                prompt_vector = build_checkpoint_prompt(input_, checkpoint, predictor, device)
                if prompt_vector is None:
                    raise ValueError("当前权重不包含 checkpoint prompt 信息，请改用 --prompt_mode fixed。")
                for prompt_idx in stage_prompt_indices:
                    restored = model([restored, typep[prompt_idx].unsqueeze(0), prompt_vector])
                    restored = torch.clamp(restored, 0, 1)
            else:
                for prompt_name in type_order:
                    restored = model([restored, typep[type_dic[prompt_name]].unsqueeze(0), multip[sce_dic[args.deg_scenario]]])
                    restored = torch.clamp(restored, 0, 1)

            restored_pil = transforms.ToPILImage()(restored[0, 0, :h, :w].detach().cpu())

        output_dir = os.path.join(out_dir, args.dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, file_name)
        save_img(output_path, restored_pil)

        if args.cal_iqa:
            score_niqe += niqe(output_path)
            score_nima += nima(output_path)
            score_musiq += musiq(output_path)

    if args.cal_iqa:
        score_niqe = score_niqe.item() / len(loader)
        score_nima = score_nima.item() / len(loader)
        score_musiq = score_musiq.item() / len(loader)
        print(f"NIQE: %.4f, NIMA: %.4f, MUSIQ: %.4f" % (score_niqe, score_nima, score_musiq))

    print("Prompt mode: %s" % prompt_mode)
    print(f"Files saved at {os.path.join(out_dir, args.dataset_name)}")
    print('finish !')


if __name__ == "__main__":
    main()
