import numpy as np
import scipy as sci
import torch
import h5py
import os
from utils.sensor_utils import get_sensors
from utils.data_utils import add_channels
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import RegularGridInterpolator


def GetFluidDataloader(n_sensor, n_steps, train_bs, test_bs, 
                        data_dir = 'path/to/data',
                        sampling_method='deim'):
    
    F = np.load(os.path.join(data_dir, 'flow_cylinder.npy'))
    F = np.concatenate((F, F[:,:,-1:]), axis=2)
    F_mean, F_std = F.mean(), F.std()
    F_train = F[0:100,:,:]
    t,m,n = F_train.shape
    F_train = F_train.reshape(100,-1) # F_train (100, 76416)
    F_test = F[100:151,:,:].reshape(51,-1) # F_test (51,76416)

    recover_factors = {'mean':F_mean, 'std':F_std}

    # get size
    outputlayer_size = F_train.shape[1] # 76416
    n_snapshots_train = F_train.shape[0] # 100
    n_snapshots_test = F_test.shape[0] # 51

    F_train = (F_train - F_mean) / F_std
    F_test = (F_test - F_mean) / F_std

    s_train, s_loc = get_sensors(F_train, sampling_method, sensor_num=n_sensor, datashape=(m, n), interestshape=(300, 80))
    s_test = F_test[:, s_loc]
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Height
    F_train_channel = add_channels(F_train) # (100, 1, 76416)
    F_test_channel = add_channels(F_test)

    # Shift data for single (multi-step)
    flow_y = F_train_channel.reshape(F_train_channel.shape[0], F_train_channel.shape[1], m, n) # [99, 1, 76416]
    flow_y_test = torch.from_numpy(F_test_channel.reshape(F_test_channel.shape[0], F_test_channel.shape[1], m, n))[n_steps:] # [50, 1, 76416]

    sensors_channel = add_channels(s_train) # 100,1,12
    sensors_test_channel = add_channels(s_test)

    s_train_input = torch.from_numpy(sensors_channel[:-n_steps])
    s_train_target, x_train_target = [], []
    for step in range(1, n_snapshots_train-n_steps+1):
        s_train_target.append(torch.from_numpy(sensors_channel[step:step+n_steps]))
        x_train_target.append(torch.from_numpy(flow_y[step:step+n_steps]))
        
    s_train_target = torch.stack(s_train_target, dim=0) 
    x_train_target = torch.stack(x_train_target, dim=0) 

    sensors_x_test = torch.from_numpy(sensors_test_channel)[0:-n_steps]
    sensors_y_test = torch.from_numpy(sensors_test_channel)[n_steps:]

    train_data = TensorDataset(s_train_input, s_train_target, x_train_target)
    train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)
    
    test_data = TensorDataset(sensors_x_test, sensors_y_test, flow_y_test)
    test_loader = DataLoader(dataset=test_data,batch_size = test_bs,shuffle=True)

    return m, n, train_loader, test_loader, s_loc, recover_factors


def GetRBCDataloader(n_sensor, n_steps, train_bs, test_bs, 
                  data_dir = 'path/to/data',
                  sampling_method='deim',
                  test_portion=.05, normalize_method='mean-std',
                  normalize_first=False, n_trajectory=1):
    print('Loading rbc data...\n')
    transform = torch.from_numpy

    with h5py.File(os.path.join(data_dir, 'rbc_diff_IC/rbc_3569_512128/rbc_3569_512128_s2.h5'), 'r') as _f:
        X = _f['tasks']['vorticity'][...]
        
    # n_trajectory = 2
    n_data = X.shape[0]
    n_train_data = int(n_data*(1-test_portion*2))
    n_val_data = int(n_data*n_trajectory*test_portion)
    
    h, w = X.shape[-2], X.shape[-1]
    h, w = 128, 128
    
    recover_factors = {'min':0., 'max':1.}
    
    X_cropped = []
    s_cropped = []
    s_locs = []
    print(n_trajectory)
    for h_ in range(n_trajectory):
        x_cropped = X[..., h_*128:(h_+1)*128, :]
        _, s_loc = get_sensors(x_cropped.reshape(x_cropped.shape[0], -1), sampling_method, sensor_num=n_sensor, datashape=(h, w))

        x_min, x_max = x_cropped.min(), x_cropped.max()
        x_mean, x_std = x_cropped.mean(), x_cropped.std()
        
        if normalize_method == 'mean-std':
            x_cropped = (x_cropped - x_mean)/ x_std
            
            recover_factors['mean'] = x_mean
            recover_factors['std'] = x_std
            recover_factors['min'] = x_min
            recover_factors['max'] = x_max
        elif normalize_method == 'max-min':
            x_cropped = (x_cropped - x_min) / (x_max - x_min)
            recover_factors['min'] = x_min
            recover_factors['max'] = x_max
        elif normalize_method == 'max-min-shift':
            x_cropped = (x_cropped - x_min) / (x_max - x_min) - .5
            
            recover_factors['min'] = x_min
            recover_factors['max'] = x_max
            recover_factors['shift'] = .5
        else:
            recover_factors['min'] = 0.
            recover_factors['max'] = 1.
            
        if normalize_first:
            s, s_loc = get_sensors(x_cropped.reshape(x_cropped.shape[0], -1), sampling_method, sensor_num=n_sensor, datashape=(h, w))
        else:
            s = x_cropped.reshape(x_cropped.shape[0], -1)[..., s_loc]
            
        s_cropped.append(np.expand_dims(s, axis=1))
        s_locs.append(s_loc)
        X_cropped.append(np.expand_dims(x_cropped, axis=1)) # [4*n_data]*1*128*128
        
        s_train_inputs, s_train_targets, x_train_targets, x_targets = [], [], [], []
        for x_train, s_train in zip(X_cropped, s_cropped):
            s_train_input = transform(s_train[:-n_steps])
            x_target = transform(x_train[n_steps:])
            s_train_target, x_train_target = [], []
            for step in range(1, n_data-n_steps+1):
                s_train_target.append(transform(s_train[step:step+n_steps]))
                x_train_target.append(transform(x_train[step:step+n_steps]))
            s_train_target = torch.stack(s_train_target, dim=0)
            x_train_target = torch.stack(x_train_target, dim=0)
            
            s_train_inputs.append(s_train_input)
            s_train_targets.append(s_train_target)
            x_train_targets.append(x_train_target)
            x_targets.append(x_target)
        s_train_inputs = torch.cat(s_train_inputs, dim=0)
        s_train_targets = torch.cat(s_train_targets, dim=0)
        x_train_targets = torch.cat(x_train_targets, dim=0)
        x_targets = torch.cat(x_targets, dim=0)
        
        s_val, s_test, X_val, X_test = s_train_inputs[-2*n_val_data:, ...], s_train_inputs[-2*n_val_data:, ...], x_targets[-2*n_val_data:, ...], x_targets[-2*n_val_data:, ...]
        s_train_inputs, s_train_targets, x_train_targets = s_train_inputs[:-2*n_val_data, ...], s_train_targets[:-2*n_val_data, ...], x_train_targets[:-2*n_val_data, ...]

    print(s_train_inputs.shape, s_train_targets.shape, x_train_targets.shape)
    train_data = TensorDataset(s_train_inputs, s_train_targets, x_train_targets)
    train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)

    print(x_targets.shape, s_val.shape, X_val.shape, s_test.shape, X_test.shape)
    s_val_input = s_val[:-n_steps]
    s_val_output = s_val[n_steps:]

    val_data = TensorDataset(s_val_input, s_val_output, X_val[n_steps:], X_val[:-n_steps])
    val_loader = DataLoader(dataset=val_data,batch_size =test_bs,shuffle=False)

    s_test_input = s_test[:-n_steps]
    s_test_output = s_test[n_steps:]

    test_data = TensorDataset(s_test_input, s_test_output, X_test[n_steps:], X_test[:-n_steps])
    test_loader = DataLoader(dataset=test_data,batch_size =test_bs,shuffle=False)
    
    sensor_locations = s_locs

    return h, w, [train_loader, val_loader], test_loader, sensor_locations, recover_factors


def GetClimateDataloader(n_sensor, n_steps, train_bs, test_bs,
                   n_channel = 2,
                   n_downscale = 8,
                   data_dir = 'path/to/data',
                   normalize_method='unnormalize',
                   sampling_method='deim'):
    
    print(f'Loading climate data...\n Channel {n_channel} Downscale {n_downscale}')
    
    transform = torch.from_numpy
    recover_factors = {'min':0., 'max':1.}

    # Xs = {}
    
    # for sep in ['train']:  #, 'val', 'test']:
    # sep_datadir = os.path.join(data_dir, 'climate', 'train')
    # fps = sorted([os.path.join(sep_datadir, fp) for fp in os.listdir(sep_datadir)])
    # data = []
    # for year, fp in enumerate(fps):
        # if year < n_year:
            # with h5py.File(fp, 'r') as _f:
                # data.append(_f.get('fields')[...][:, n_channel, :])
    # data = np.concatenate(data, axis=0)

    if n_downscale in [4, 8]:
        datadir = os.path.join(data_dir, 'climate', 'prep_data', f'climate_ds{n_downscale}_c{n_channel}.h5')
        with h5py.File(datadir, 'r') as f:
            data = f['fields'][()]
    else:
        whole_map_by_channel_pth = os.path.join(data_dir, 'climate', f'climate_whole_map_c{n_channel}.h5')
        whole_map_by_channel = h5py.File(whole_map_by_channel_pth, 'r')['fields'][()]
        data = climate_downsample(whole_map_by_channel, 0, n_downscale)
        print(f'shape after downsample of data is {data.shape}')

    h, w = data.shape[-2], data.shape[-1]
    data = data.reshape(data.shape[0], -1)

    recover_factors['min'], recover_factors['max'] = data.min(), data.max()
    
    X = data

    if normalize_method == 'mean-std':
        recover_factors['mean'], recover_factors['std'] = data.mean(), data.std()
    elif normalize_method == 'px-mean-std':
        recover_factors['mean'], recover_factors['std'] = data.mean(axis=0), data.std(axis=0)
    elif normalize_method == 'unnormalize':
        recover_factors['mean'], recover_factors['std'] = 0., 1.
            
    print(normalize_method)
    
    X = (X-recover_factors['mean'])/recover_factors['std']
    
    s, sensor_loc = get_sensors(X, sampling_method, sensor_num=n_sensor, datashape=(h, w))
        
    X_train, X_val, X_test = X[:-2*365, ...], X[-2*365:-1*365, ...], X[-1*365:, ...]
    s_train, s_val, s_test = s[:-2*365, ...], s[-2*365:-1*365, ...], s[-1*365:, ...]    
    print(s_train.shape, X_train.shape, s_val.shape, X_val.shape, s_test.shape, X_test.shape)
    
    n_train_data = X_train.shape[0]

    # add channel
    X_train = X_train[:, np.newaxis, :] # [n_x_train, 1, h*w]
    X_test = X_test[:, np.newaxis, :]   # [n_x_test, 1, h*w]
    X_val = X_val[:, np.newaxis, :]   # [n_x_test, 1, h*w]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], h, w) # [n_x_train, 1, h, w]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], h, w)[n_steps:] # [n_x_test-n_steps, 1, h, w]
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], h, w)[n_steps:] # [n_x_test-n_steps, 1, h, w]

    s_train = s_train[:, np.newaxis, :] # [n_s_train, 1, n_sensors]
    s_test = s_test[:, np.newaxis, :] # [n_s_test, 1, n_sensors]
    s_val = s_val[:, np.newaxis, :] # [n_s_val, 1, n_sensors]

    # List[Torch.Tensor] - [(n_steps + 1) * Torch.Size(n_s_train - n_steps, 1, n_sensors)]
    s_train_input = transform(s_train[:-n_steps])
    # x_target = transform(X_train[n_steps:])
    s_train_target, x_train_target = [], []
    for step in range(1, n_train_data-n_steps+1):
        s_train_target.append(transform(s_train[step:step+n_steps]))
        x_train_target.append(transform(X_train[step:step+n_steps]))
    s_train_target = torch.stack(s_train_target, dim=0)
    x_train_target = torch.stack(x_train_target, dim=0)
    
    # [n_s_test-n_steps, 1, n_sensors] starts from s_test_idx=0
    s_val_input = torch.from_numpy(s_val)[:-n_steps]
    # [n_s_test-n_steps, 1, n_sensors] starts from s_test_idx=n_steps
    s_val_output = torch.from_numpy(s_val)[n_steps:]

    # [n_s_test-n_steps, 1, n_sensors] starts from s_test_idx=0
    s_test_input = torch.from_numpy(s_test)[:-n_steps]
    # [n_s_test-n_steps, 1, n_sensors] starts from s_test_idx=n_steps
    s_test_output = torch.from_numpy(s_test)[n_steps:]

    train_data = TensorDataset(s_train_input, s_train_target, x_train_target)
    train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)
    
    val_data = TensorDataset(s_val_input, s_val_output, transform(X_val))
    val_loader = DataLoader(dataset=val_data,batch_size =test_bs,shuffle=False)

    test_data = TensorDataset(s_test_input, s_test_output, transform(X_test))
    test_loader = DataLoader(dataset=test_data,batch_size =test_bs,shuffle=False)
    
    # print(s_val_input.shape, s_val_output.shape, X_val.shape)

    m, n = h, w
    sensor_locations = sensor_loc

    json_recover_factores = {}
    for k, v in recover_factors.items():
        json_recover_factores[k] = np2jsonEncoder(v)
    print(json_recover_factores)

    return m, n , [train_loader, val_loader], test_loader, sensor_locations, json_recover_factores


def GetErrorPropDataloader(tdata='fluid', prop_steps=range(3, 28, 3), **kwargs):
    test_loaders = []
    tdata2func = {'fluid':GetFluidDataloader}
    for prop_step in prop_steps:
        dataloader = tdata2func[tdata]
        _, _, _, test_loader, _, recover_factors = dataloader(n_steps=prop_step, **kwargs)
        test_loaders.append(test_loader)
    return test_loaders, recover_factors


def np2jsonEncoder(npvals):
    if isinstance(npvals, np.integer):
        return int(npvals)
    elif isinstance(npvals, np.floating):
        return float(npvals)
    elif isinstance(npvals, np.ndarray):
        return npvals.tolist()
    else:
        return npvals


def climate_downsample(data, ds_t=0, ds_hw=8, interp_method='linear'):
    if len(data.shape) == 3:
        sigma = [ds_t//4, ds_hw//8, ds_hw//8]  #[ds_t//2, ds_hw//2, ds_hw//2]
    elif len(data.shape) == 4:  
        # assume data shape is [t, c, h, w]
        sigma = [ds_t//2, 0, ds_hw//2, ds_hw//2]
    gaussian_filtered_data = sci.ndimage.gaussian_filter(data, sigma=sigma)
    
    interp = RegularGridInterpolator(
        tuple([np.arange(s) for s in list(gaussian_filtered_data.shape)]),
        values=gaussian_filtered_data, method=interp_method
    )


    ds_lst = [ds_factor if ds_factor != 0 else 1 for ds_factor in [ds_t, ds_hw, ds_hw]]

    if len(data.shape) == 4:
        data = data.transpose(0, 2, 3, 1)

    meshgrid_list = [np.linspace(0, gaussian_filtered_data.shape[ds_idx]-1, 
                        gaussian_filtered_data.shape[ds_idx]//ds_factor)
                        for ds_idx, ds_factor in enumerate(ds_lst)]
    meshgrid = np.meshgrid(*meshgrid_list, indexing='ij')
    lres_coord = np.stack(meshgrid, axis=-1)

    if len(data.shape) == 4:
        lres_coord = lres_coord.transpose(0, 3, 1, 2)

    space_time_crop_lres = interp(lres_coord)
    return space_time_crop_lres
