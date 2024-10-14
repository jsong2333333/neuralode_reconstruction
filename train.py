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
from models.model import fno_baseline
from models.model import encoder_node_decoder


MODEL_HYPERPARAMS = {}
PRETRAIN = False

def train(model, loaders, criterion, optimizer, scheduler, n_steps, rho, epochs, recover_factors, neptune_run, save_pth='', curr_epoch=0, model_arch='gnode'):
    best_loss = 1e8
    best_recon_err = 1e8
    start2 = time.time()

    train_loader, val_loader = loaders
    
    # switch to train mode
    model.train()
    for epoch in range(curr_epoch, epochs):
        model.train()
        start = time.time()
        dynamic_losses, recon_losses, total_losses = 0., 0., 0.
        n = 0

        for _,(s_train_input, s_train_target, x_train_target) in enumerate(train_loader):
            if model_arch == 'gnode':
                s_next, x_next = model(s=s_train_input.float().to(DEVICE), hi_res_result=True)
                dynamic_loss   = criterion(s_next, s_train_target[:, 0, :].float().to(DEVICE))
                recon_loss     = criterion(x_next, x_train_target[:, 0, :].float().to(DEVICE))
            elif model_arch == 'fno':
                x_next = model(s=s_train_input.float().to(DEVICE), hi_res_result=True)
                recon_loss     = criterion(x_next, x_train_target[:, 0, :].float().to(DEVICE))
            elif model_arch == 'mae':
                x_next, mask = model(X=x_train_target[:, 0, :].float().to(DEVICE), hi_res_result=False)
                # dynamic_loss   = criterion(s_next, (x_train_target[:, 0, :]*(1-mask)).float().to(DEVICE))
                recon_loss     = criterion(x_next*mask, x_train_target[:, 0, :].float().to(DEVICE)*mask)

            for step in range(1, n_steps):
                if model_arch == 'gnode':
                    s_next, x_next = model(s=s_next, hi_res_result=True)
                    dynamic_loss  += criterion(s_next, s_train_target[:, step, :].float().to(DEVICE))
                    recon_loss    += criterion(x_next, x_train_target[:, step, :].float().to(DEVICE))
                elif model_arch == 'fno':
                    x_next = model(s=x_next.float().to(DEVICE), hi_res_result=False)
                    recon_loss    += criterion(x_next, x_train_target[:, step, :].float().to(DEVICE))
                elif model_arch == 'mae':
                    x_next, mask = model(X=x_train_target[:, step, :].float().to(DEVICE), hi_res_result=False)
                    # dynamic_loss  += criterion(s_next, (x_train_target[:, step, :]*(1-mask)).float().to(DEVICE))
                    recon_loss    += criterion(x_next*mask, x_train_target[:, step, :].float().to(DEVICE)*mask)
                            
            if model_arch == 'gnode':
                loss =  rho * dynamic_loss + recon_loss
                dynamic_losses += dynamic_loss.item()
            else:
                loss = recon_loss
            recon_losses   += recon_loss.item()
            total_losses   += loss.item()

            # ================== backward ===================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += s_train_input.shape[0]

        scheduler.step()

         # record train loss
        train_loss_mean = total_losses / n
        # neptune_run.setdefault('train/train_loss', []).append(total_losses / n)
        # neptune_run.setdefault('train/dynamic_loss', []).append(dynamic_losses / n)
        # neptune_run.setdefault('train/reconstruction_loss', []).append(recon_losses / n)
        neptune_run['train/train_loss'].append(total_losses / n)
        if model_arch == 'gnode':
            neptune_run['train/dynamic_loss'].append(dynamic_losses / n)
        neptune_run['train/reconstruction_loss'].append(recon_losses / n)

        print(f'Epoch {epoch}: dynamic loss - {dynamic_losses / n}; reconstruction loss - {recon_losses / n}; train loss - {total_losses / n}')

        if train_loss_mean <= best_loss:
            best_loss = train_loss_mean
            torch.save({
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'model_hyperparams': MODEL_HYPERPARAMS
                }, os.path.join(save_pth, 'checkpoint.pt'))
            
        end = time.time()
        # neptune_run.setdefault('train/train_epoch_time', []).append(end - start)
        neptune_run['train/train_epoch_time'].append(end - start)

        # if (epoch % 10 == 9 or epoch == 0) and val_loader is not None:
        #     err_lst, _, _ = eval_or_test(model, val_loader, n_steps, recover_factors, neptune_run, eval_or_test = 'val', save_pth=save_pth, model_arch=model_arch)
        #     # neptune_run.setdefault('val/val_time_per_20_epoch', []).append(val_time)

        #     # for k, v in err_lst.items():
        #     #     print(f'{k} - {np.mean(v)}')
        #     mean_norm_recon_err = np.mean(err_lst['norm_val_recon_err'])
        #     if mean_norm_recon_err <= best_recon_err:
        #         best_recon_err = mean_norm_recon_err
        #         torch.save({
        #             'model_state_dict': model.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'epoch': epoch,
        #             'model_hyperparams': MODEL_HYPERPARAMS
        #             }, os.path.join(save_pth, 'checkpoint_best_val.pt'))

    end2 = time.time()
    print('The training time is: ', (end2 - start2))
    neptune_run['train/final_train_time'] = end2 - start2


def eval_or_test(model, test_loader, n_steps, recover_factors, neptune_run, eval_or_test = 'test', save_pth='', model_arch='gnode'):
    print(f'---- {eval_or_test.upper()} -----')

    if eval_or_test == 'val':
        model.eval()
    else:
        try:
            checkpoint = torch.load(os.path.join(save_pth, 'checkpoint_best_val.pt'))
        except:
            checkpoint = torch.load(os.path.join(save_pth, 'checkpoint.pt'))
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        model.eval()

    err_lst = {f'norm_{eval_or_test}_dym_err':[],
               f'norm_{eval_or_test}_recon_err':[],
               f'unnorm_{eval_or_test}_dym_err':[],
               f'unnorm_{eval_or_test}_recon_err':[]}

    for _, data in enumerate(test_loader):
        s_input, s_target, x_target = data

        if model_arch == 'gnode':
            output_y, output_Y = model(s=s_input.float().to(DEVICE), hi_res_result=True)
        else:
            output_Y, *_ = model(s=s_input.float().to(DEVICE), hi_res_result=True)
        k = 0
        while k < n_steps-1:
            if model_arch == 'gnode':
                if k == n_steps - 2:
                    output_y, output_Y = model(s=output_y, hi_res_result=True)
                else:
                    output_y, _ = model(s=output_y, hi_res_result=False)
                k+=1
            elif model_arch == 'fno':
                output_Y = model(s=output_Y, hi_res_result=False)
            else:
                output_Y, _ = model(s=output_Y, hi_res_result=True)

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
                output_Y = output_Y.detach().cpu()

                if n == 'norm':
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
                    for p in [output_Y, x_target]:
                        p += recover_shift
                        p *= recover_range
                        p += recover_addition
                        unnomalized_lst.append(p)
                    output_Y, x_target = unnomalized_lst

                    err2 = (torch.norm(output_Y - x_target, p=2, dim=(1, 2, 3)) / torch.norm(x_target, p=2, dim=(1, 2, 3))).mean().item()

                target_Y = x_target
                err_lst[f'{n}_{eval_or_test}_recon_err'].append(err2)

        # print(err_lst)
        
    for k, v in err_lst.items():
        # neptune_run.setdefault(f'{eval_or_test}/{k}', []).append(np.mean(v))
        neptune_run[f'{eval_or_test}/{k}'].append(np.mean(v))

    if eval_or_test == 'val':
        model.train()
    else:
        for k, v in err_lst.items():
            print(f'{k} - {np.mean(v)}')

    return err_lst, target_Y, output_Y
    

def main():
    set_seed(RANDOM_SEED)
    
    run = neptune.init_run(
        project = "project_name",
        api_token = "neptune_token"
    )
    run_id = run['sys/id'].fetch()
    
    if args.pretrain and os.path.exists(args.pretrain_pth):
        save_dir = os.path.join(os.path.dirname(args.pretrain_pth), 'pretrain')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) 
        model_ckpt = torch.load(args.pretrain_pth)
        PRETRAIN = True
    else:
        PRETRAIN = False
        save_dir = f'./{args.model_id}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    curr_time = datetime.now().strftime("%m-%d-%H-%M-%S")
    save_pth = os.path.join(save_dir, f'./{run_id}_{curr_time}')
    os.mkdir(save_pth)
    
    if TDATA == 'climate':
        h, w, train_val_loaders, test_loader, sensor_locations, recover_factors = GetClimateDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            n_channel=N_CHANNEL,
                                                            data_dir='path/to/data',
                                                            normalize_method=NORMALIZE_METHOD,
                                                            sampling_method=SAMPLING)
    elif TDATA == 'rbc':
        h, w, train_val_loaders, test_loader, sensor_locations, recover_factors = GetRBCDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            sampling_method=SAMPLING,
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            data_dir='path/to/data',
                                                            test_portion=.05,
                                                            normalize_first=NORMALIZE_FIRST,
                                                            normalize_method=NORMALIZE_METHOD)
    elif TDATA == 'fluid':
        h, w, train_loader, test_loader, sensor_locations, recover_factors = GetFluidDataloader(n_sensor=N_SENSOR, 
                                                            n_steps=N_STEP, 
                                                            sampling_method=SAMPLING,
                                                            train_bs=TRAIN_BATCH_SIZE, 
                                                            test_bs=TEST_DATA_SIZE,
                                                            data_dir='path/to/data')
        train_val_loaders = [train_loader, test_loader]

    for k, v in recover_factors.items():
        recover_factors[k] = float(v)
    
    if not PRETRAIN:
        np.save(os.path.join(save_pth, 'sensor_locations.npy'), sensor_locations)
        json.dump(recover_factors, open(os.path.join(save_pth, 'recover_factors.json'), 'w'))

        
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
    if PRETRAIN:
        model_state_dict = {}
        for k, v in model_ckpt['model_state_dict'].items():
            new_k = k
            if new_k.startswith('module'):
                new_k = new_k[7:]
            model_state_dict[new_k] = v
            
        model.load_state_dict(model_state_dict)
        
        if 'epoch' in model_ckpt:
            curr_epoch = model_ckpt['epoch']
        
    model.to(DEVICE)
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    run['parameters'] =  MODEL_HYPERPARAMS
    
    optimizer = optim.Adam(model.parameters(), lr= LR, betas= (0.9,0.999), eps= 1e-8, weight_decay= WD)
    if LR_SCHEDULER == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
    elif LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.5)
        
    if PRETRAIN:
        scheduler.load_state_dict(model_ckpt['scheduler_state_dict'])

    criterion = nn.MSELoss()

    train(model,
            train_val_loaders,
            criterion,
            optimizer,
            scheduler,
            n_steps=N_STEP,
            rho=RHO,
            epochs=N_EPOCH,
            recover_factors=recover_factors,
            neptune_run=run,
            save_pth=save_pth,
            curr_epoch=curr_epoch,
            model_arch=args.model_arch)
    
    rel_err_lst, target_Y, output_Y = eval_or_test(model, 
                                                    test_loader, 
                                                    n_steps=N_STEP, 
                                                    recover_factors=recover_factors, 
                                                    neptune_run=run,
                                                    eval_or_test='test',
                                                    save_pth=save_pth,
                                                    model_arch=args.model_arch)
    
    json.dump(rel_err_lst, open(os.path.join(save_pth, 'rel_err_lst.json'), 'w'))
    np.save(os.path.join(save_pth, 'target_Y.npy'), target_Y.detach().cpu().numpy())
    np.save(os.path.join(save_pth, 'output_Y.npy'), output_Y.detach().cpu().numpy())
    
    run.stop()


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
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_pth', default='', type=str)
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
    PRETRAIN = args.pretrain
    PRETRAIN_PATH = args.pretrain_pth
    N_CHANNEL = args.n_channel
    NORMALIZE_METHOD = args.normalize_method
    LR_SCHEDULER = args.lr_scheduler
    N_DOWNSCALE = args.n_downscale
    RANDOM_SEED = args.random_seed
    TNODE = args.tnode
    TACTIVATION = args.tactivation
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_HYPERPARAMS = {k.upper():v for k, v in vars(args).items()}
    main()
