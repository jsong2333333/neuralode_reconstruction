import torch
from torch import nn
import torch.optim as optim
import numpy as np
from utils.dataloader import *
from models.model import node_decoder
from utils.data_utils import set_seed
import os
import time
from datetime import datetime
import json
import argparse
import neptune
from neptune_token import NEPTUNE_API_TOKEN
from models.model import fno_baseline, encoder_node_decoder


MODEL_HYPERPARAMS = {}

def eval_or_test(model, test_loader, n_steps, recover_factors, neptune_run, eval_or_test = 'test', save_pth='', model_arch='gnode'):
    print(f'---- {eval_or_test.upper()} -----')
    model.eval()

    err_lst = {f'norm_{eval_or_test}_dym_err':[],
               f'norm_{eval_or_test}_recon_err':[],
               f'unnorm_{eval_or_test}_dym_err':[],
               f'unnorm_{eval_or_test}_recon_err':[]}

    # with torch.no_grad():
    for idx_data, data in enumerate(test_loader):
        print(idx_data)
        s_input, s_target, x_target, x_ini = data

        if model_arch == 'gnode':
            output_y, output_Y = model(s=s_input.float().to(DEVICE), hi_res_result=True)
        elif model_arch == 'fno':
            output_Y = model(s=s_input.float().to(DEVICE), hi_res_result=True)
        elif model_arch == 'mae':
            output_Y, _ = model(X=x_ini.float().to(DEVICE), hi_res_result=True)

        for _ in range(1, n_steps):
            if model_arch == 'gnode':
                output_y, output_Y = model(s=output_y, hi_res_result=True)
            elif model_arch == 'fno':
                output_Y = model(s=output_Y, hi_res_result=False)
            elif model_arch == 'mae':
                output_Y, _ = model(X=output_Y, hi_res_result=True)


        if model_arch == 'gnode':
            for n in ['norm', 'unnorm']:
                output_y = output_y.detach().cpu()
                output_Y = output_Y.detach().cpu()

                if n == 'norm':
                    err1 = (torch.norm(output_y - s_target, p=2, dim=(1, 2)) / torch.norm(s_target, p=2, dim=(1, 2))).mean().item()
                    err2 = (torch.norm(output_Y - x_target, p=2, dim=(1, 2, 3)) / torch.norm(x_target, p=2, dim=(1, 2, 3))).mean().item()
                else:
                    recover_shift = 0.
                    if 'mean' in recover_factors.keys():
                        recover_addition, recover_range = recover_factors['mean'], recover_factors['std']
                    elif 'shift' in recover_factors.keys():
                        recover_addition, recover_range = recover_factors['min'], recover_factors['max'] - recover_factors['min']
                        recover_shift = recover_factors['shift']
                    else:
                        recover_addition, recover_range = recover_factors['min'], recover_factors['max'] - recover_factors['min']

                    unnomalized_lst = []    
                    for p in [output_y, output_Y, s_target, x_target]:
                        p += recover_shift
                        p *= recover_range
                        p += recover_addition
                        unnomalized_lst.append(p)
                    output_y, output_Y, s_target, x_target = unnomalized_lst

                    err1 = (torch.norm(output_y - s_target, p=2, dim=(1, 2)) / torch.norm(s_target, p=2, dim=(1, 2))).mean().item()
                    err2 = (torch.norm(output_Y - x_target, p=2, dim=(1, 2, 3)) / torch.norm(x_target, p=2, dim=(1, 2, 3))).mean().item()

                target_Y = x_target

                err_lst[f'{n}_{eval_or_test}_dym_err'].append(err1)
                err_lst[f'{n}_{eval_or_test}_recon_err'].append(err2)
        else:
            for n in ['norm', 'unnorm']:
                if n == 'norm':
                    err2 = (torch.norm(output_Y - x_target.cuda(), p=2, dim=(1, 2, 3)) / torch.norm(x_target.cuda(), p=2, dim=(1, 2, 3))).mean().cpu().item()
                else:
                    recover_shift = 0.
                    if 'mean' in recover_factors.keys():
                        recover_addition, recover_range = recover_factors['mean'], recover_factors['std']
                    elif 'shift' in recover_factors.keys():
                        recover_addition, recover_range = recover_factors['min'], recover_factors['max'] - recover_factors['min']
                        recover_shift = recover_factors['shift']
                    else:
                        recover_addition, recover_range = recover_factors['min'], recover_factors['max'] - recover_factors['min']

                    unnomalized_lst = []    
                    for p in [output_Y, x_target]:
                        p += recover_shift
                        p *= recover_range
                        p += recover_addition
                        unnomalized_lst.append(p)
                    output_Y, x_target = unnomalized_lst

                    err2 = (torch.norm(output_Y - x_target.cuda(), p=2, dim=(1, 2, 3)) / torch.norm(x_target.cuda(), p=2, dim=(1, 2, 3))).mean().cpu().item()

                target_Y = x_target
                err_lst[f'{n}_{eval_or_test}_recon_err'].append(err2)

            del s_input, s_target, x_target

    for k, v in err_lst.items():
        print(f'{k} - {np.mean(v)}')

    return err_lst, target_Y, output_Y
    

def main():
    set_seed(RANDOM_SEED)
    
    save_dir = os.path.dirname(args.eval_pth)

    if TDATA == 'climate':
        h, w, train_val_loaders, test_loader, sensor_locations, recover_factors = GetClimateDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            n_channel=N_CHANNEL,
                                                            data_dir='path/to/data/',
                                                            normalize_method=NORMALIZE_METHOD,
                                                            sampling_method=SAMPLING)
    elif TDATA == 'rbc':
        h, w, train_val_loaders, test_loader, sensor_locations, recover_factors = GetRBCDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            sampling_method=SAMPLING,
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            data_dir='path/to/data/',
                                                            test_portion=.05,
                                                            normalize_first=NORMALIZE_FIRST,
                                                            normalize_method=NORMALIZE_METHOD)
    elif TDATA == 'fluid':
        h, w, train_loader, test_loader, sensor_locations, recover_factors = GetFluidDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            sampling_method=SAMPLING,
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            data_dir='path/to/data/')
        train_val_loaders = [train_loader, test_loader]

    for k, v in recover_factors.items():
        recover_factors[k] = float(v)
        
    for k, v in recover_factors.items():
        MODEL_HYPERPARAMS[f'DATA_NORM_FACTOR_{k}'] = v

    # run = {}
    
    if args.model_arch == 'gnode':
        model = node_decoder(hid_feats_node=NODE_H_FE,
                                n_hid_layers_node=NODE_N_H_LAYER,
                                n_sensors=N_SENSOR,
                                high_res=[h, w],
                                up_factor=UP_FACTOR,
                                hid_feats_trans=TRANS_H_FE,
                                window_size=WINDOW_SIZE,
                                epsilon_node=EPS,
                                substeps_node=SUBSTEPS,
                                tnode=TNODE,
                                tactivation=TACTIVATION)
    elif args.model_arch == 'fno':
        model = fno_baseline(hid_feats_node=NODE_H_FE,
                                n_hid_layers_node=NODE_N_H_LAYER,
                                n_sensors=N_SENSOR,
                                high_res=[h, w],
                                up_factor=UP_FACTOR,
                                hid_feats_trans=TRANS_H_FE)
    elif args.model_arch == 'mae':
        model = encoder_node_decoder(hid_feats_node=NODE_H_FE,
                                n_hid_layers_node=NODE_N_H_LAYER,
                                n_sensors=N_SENSOR,
                                high_res=[h, w],
                                up_factor=UP_FACTOR,
                                hid_feats_trans=TRANS_H_FE,
                                window_size=WINDOW_SIZE,
                                epsilon_node=EPS,
                                substeps_node=SUBSTEPS,
                                tnode=TNODE,
                                tactivation=TACTIVATION)
    print(model)

    curr_epoch = 0
    model_state_dict = {}
    model_ckpt = torch.load(args.eval_pth, map_location=torch.device('cpu'))
    for k, v in model_ckpt['model_state_dict'].items():
        new_k = k
        if new_k.startswith('module'):
            new_k = new_k[7:]
        model_state_dict[new_k] = v
        
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    
    rel_err_lst, target_Y, output_Y = eval_or_test(model, 
                                            test_loader, 
                                            n_steps=N_STEP, 
                                            recover_factors=recover_factors, 
                                            neptune_run=None,
                                            eval_or_test='test',
                                            save_pth=save_dir,
                                            model_arch=args.model_arch)
    
    json.dump(rel_err_lst, open(os.path.join(save_dir, 'rel_err_lst.json'), 'w'))
    np.save(os.path.join(save_dir, 'target_Y.npy'), target_Y.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'output_Y.npy'), output_Y.detach().cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLOW-RECON')
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--test_data_size', default=50, type=int)
    parser.add_argument('--n_sensor', default=16, type=int)
    parser.add_argument('--n_step', default=15, type=int)
    parser.add_argument('--normalize_first', action="store_true", default=False)
    parser.add_argument('--node_h_fe', default=64, type=int)
    parser.add_argument('--trans_h_fe', default=64, type=int)
    parser.add_argument('--node_n_h_layer', default=3, type=int)
    parser.add_argument('--up_factor', default=4, type=int)
    parser.add_argument('--n_epoch', default=400, type=int)
    parser.add_argument('--window_size', default=2, type=int)
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--eps', default=1., type=float)
    parser.add_argument('--substeps', default=4, type=int)
    parser.add_argument('--rho', default=.8, type=float)
    parser.add_argument('--lr', default=8.5e-4, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)
    parser.add_argument('--sampling', default='deim', type=str)
    parser.add_argument('--model_id', default='', type=str)
    parser.add_argument('--tdata', default='fluid', type=str)
    parser.add_argument('--n_downscale', default=8, type=int)
    parser.add_argument('--eval_pth', default='', type=str)
    parser.add_argument('--n_channel', default=2, type=int)
    parser.add_argument('--normalize_method', default='mean-std', type=str)
    parser.add_argument('--lr_scheduler', default='exp', type=str)  #['exp', 'step']
    parser.add_argument('--random_seed', default=24678, type=int)
    parser.add_argument('--tnode', default='gnode2', type=str)
    parser.add_argument('--tactivation', default='rational', type=str)
    parser.add_argument('--model_arch', default='gnode', type=str)
    args = parser.parse_args()

    '''
    HyperParams Setting for Network
    '''
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_DATA_SIZE = args.test_data_size
    N_SENSOR = args.n_sensor
    N_STEP = args.n_step
    NORMALIZE_FIRST = args.normalize_first
    UP_FACTOR = args.up_factor
    N_EPOCH = args.n_epoch
    WINDOW_SIZE = args.window_size
    CROP_SIZE = args.crop_size
    EPS = args.eps
    RHO = args.rho
    LR = args.lr 
    WD = args.wd
    SAMPLING = args.sampling
    MODEL_ID = args.model_id
    TDATA = args.tdata
    NODE_H_FE = args.node_h_fe
    NODE_N_H_LAYER = args.node_n_h_layer
    TRANS_H_FE = args.trans_h_fe
    SUBSTEPS = args.substeps
    EVAL_PATH = args.eval_pth
    N_CHANNEL = args.n_channel
    NORMALIZE_METHOD = args.normalize_method
    LR_SCHEDULER = args.lr_scheduler
    N_DOWNSCALE = args.n_downscale
    RANDOM_SEED = args.random_seed
    TNODE = args.tnode
    TACTIVATION = args.tactivation
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    HyperParams Setting for Network
    '''
    MODEL_HYPERPARAMS = {k.upper():v for k, v in vars(args).items()}
    main()
