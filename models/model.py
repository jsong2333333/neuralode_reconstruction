import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from models.decoder import SwinIR, ShallowDecoder
from models.activation import Rational
import torchvision.transforms as v2
from models.fno import fno
from models.vit_mae import MAE
import numpy as np
import copy
from timm.models.vision_transformer import PatchEmbed, Block


def get_activation(name='rational'):
    if name == 'rational':
        return Rational()
    elif name == 'relu':
        return torch.nn.ReLU()
    elif name == 'sigmoid':
        return torch.nn.Sigmoid()
    

def relu_net(input_features,hidden_layers,hidden_neurons,output_features):
    layers = []

    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_features,hidden_neurons))
        layers.append(torch.nn.ReLU())
        input_features = hidden_neurons

    layers.append(nn.Linear(input_features,output_features))
    return nn.Sequential(*layers)


def rational_net(input_features,hidden_layers,hidden_neurons,output_features):
    layers = []

    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_features,hidden_neurons))
        layers.append(Rational())
        input_features = hidden_neurons

    layers.append(nn.Linear(input_features,output_features))
    return nn.Sequential(*layers)


def nnet(input_features,hidden_layers,hidden_neurons,output_features,t_activation):
    layers = []

    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_features,hidden_neurons))
        layers.append(get_activation(t_activation))
        input_features = hidden_neurons

    layers.append(nn.Linear(input_features,output_features))
    return nn.Sequential(*layers)


def trans_net(in_feats, hid_feats, out_feats):
    '''
    Transition model using MLP: change latent variable s_{t+1} -> z_{t+1}
    The shape of s_{t+1}: [b, 1, n_sensors]
    The shape of z_{t+1}: [b, 1, (h/r) * (w/r)]

    Args: 
    -----
    in_feats: int, the number of sparse sensors (n_sensors)
    hid_feats: list, the list of hidden channels, e.g., [128, 128]
    out_feats: int, (h/r) * (w/r)
    '''

    trans_layers = []
    n_hid_layers = len(hid_feats)

    for i in range(n_hid_layers):
        trans_layers.append(nn.Linear(in_feats, hid_feats[i]))
        # trans_layers.append(nn.BatchNorm1d(hid_feats[i])) # TODO: maybe remove batchnorm
        trans_layers.append(nn.ReLU(True)) 
        in_feats = hid_feats[i]

    trans_layers.append(nn.Linear(in_feats, out_feats))

    return nn.Sequential(*trans_layers)


class LEMCell(nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        print(transformed_inp.shape)
        print(transformed_hid.shape)
        
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 0)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 0)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)

        return y, z


class LEM(nn.Module):
    def __init__(self, ninp, nhid, nlayers=1,  dt=1., steps=None):
        super(LEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp,nhid,dt)
        self.classifier = nn.Linear(nhid, ninp)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input, y=None, z=None):
        ## initialize hidden states
        if y is None:
            y = input.data.new(input.size(0), self.nhid).zero_()
            z = input.data.new(input.size(0), self.nhid).zero_()
        
        y, z = self.cell(input,y,z)
        out = self.classifier(y)
        return out, y, z


class node(nn.Module):
    '''
    Basic Neural ODE method
    '''
    def __init__(self, in_feats, hid_feats, n_hid_layers, epsilon, substeps, t_activation):
        super(node, self).__init__()
        self.NN = nnet(in_feats, n_hid_layers, hid_feats, in_feats, t_activation)
        self.epsilon = epsilon

    def forward(self, s):
        s_next = s + self.epsilon * self.NN(s)
        return s_next


class gnode2(nn.Module):
    def __init__(self, in_feats, hid_feats, n_hid_layers, epsilon, substeps, t_activation):
        super(gnode2, self).__init__()
        self.NN1 = nnet(in_feats, n_hid_layers, hid_feats, in_feats, t_activation)
        self.G1 = nnet(in_feats, n_hid_layers, hid_feats, in_feats, t_activation)
        self.epsilon = epsilon
    
    def forward(self, s):
        z = self.NN1(s)
        g = self.epsilon * torch.sigmoid(self.G1(s))
        s_next = (1-g) * s + g * z
        return s_next
    

class node_decoder(nn.Module):
    '''
    Combine NeuralODE and decoder to predict high-resolution dynamics
        CNODE: learn dynamics on sparse sensor measurements, s_t -> s_{t+1}
        Decoder: upsample to high-resolution dynamics, s_{t+1} -> x_{t+1}
    '''
    def __init__(self, hid_feats_node, n_hid_layers_node, 
                 hid_feats_trans, n_sensors, high_res, up_factor=8, window_size=8, epsilon_node=1, substeps_node=10,
                 tnode='gnode2', tactivation='rational'):
        '''
        Args: 
        -----
            hid_feats_node: int, 
                            the hidden features of NeuralODE model
            hid_feats_decoder: int,
                            the hidden features of decoder model
            n_hid_layers_node: int,
                            the number of hidden layers used for NeuralODE model
        '''


        super(node_decoder, self).__init__()

        self.hid_feats_node = hid_feats_node
        # self.hid_feats_decoder = hid_feats_decoder
        self.n_hid_layers_node = n_hid_layers_node
        # self.n_res_blocks_decoder = n_res_blocks_decoder
        self.epsilon_node = epsilon_node
        self.substeps_node = substeps_node
        self.h, self.w = high_res
        self.n_sensors = n_sensors
        self.up_factor = up_factor
        self.window_size = window_size
        self.hid_feats_trans = hid_feats_trans

        # Neural ODE model, s_t -> s_{t+1}, s variables shape: [b, c=1, n_sensors]
        if tnode == 'node':
            self.node = node(in_feats=self.n_sensors, hid_feats=self.hid_feats_node, n_hid_layers=self.n_hid_layers_node, epsilon=self.epsilon_node, substeps=self.substeps_node, t_activation=tactivation)
        elif tnode == 'gnode2':
            self.node = gnode2(in_feats=self.n_sensors, hid_feats=self.hid_feats_node, n_hid_layers=self.n_hid_layers_node, epsilon=self.epsilon_node, substeps=self.substeps_node, t_activation=tactivation)
       
        # transition model, s_{t+1} -> z_{t+1}, z variable shape: [b, 1, (h/r) * (w/r)]
        self.trans_h, self.trans_w = int(self.h / self.up_factor), int(self.w / self.up_factor)
        out_feats = self.trans_h * self.trans_w
        self.transition = trans_net(in_feats=self.n_sensors, hid_feats=[self.hid_feats_trans,self.hid_feats_trans], out_feats=out_feats) 

        # decoder model
        self.decoder = SwinIR(img_size=(self.trans_h, self.trans_w), upscale=self.up_factor, in_chans=1, window_size=self.window_size,
                                img_range=1., mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', embed_dim=60,
                                depths= [6, 6, 6, 6],  
                                num_heads= [6, 6, 6, 6])
        

    def forward(self, s, hi_res_result=False):
        '''
        Args:
        -----
        s: torch tensor, shape: [b, c=1, n_sensors] 
           sparse sensor measurement data
        epsilon: float, a hyper-parameter for tuning NODE model
        '''
        # NODE: s_t -> s_{t+1}
        s_next = self.node(s)

        x_next = None

        if hi_res_result:
            # Transition: s_{t+1}, [b, c=1, n_sensors] -> z_{t+1}, [b, 1, (h/r) * (w/r)]
            x_next = self.transition(s_next)

            # Transition: reshape to [b, 1, (h/r), (w/r)]
            x_next = x_next.view(x_next.shape[0], x_next.shape[1], int(self.h / self.up_factor), int(self.w / self.up_factor))

            x_next = self.decoder(x_next)
        return s_next, x_next 
    

class encoder_node_decoder(nn.Module):
    '''
    Combine NeuralODE and decoder to predict high-resolution dynamics
        CNODE: learn dynamics on sparse sensor measurements, s_t -> s_{t+1}
        Decoder: upsample to high-resolution dynamics, s_{t+1} -> x_{t+1}
    '''
    def __init__(self, hid_feats_node, n_hid_layers_node, 
                 hid_feats_trans, n_sensors, high_res, up_factor=8, window_size=8, epsilon_node=1, substeps_node=10,
                 tnode='gnode2', tactivation='rational', encoder_embed_dim=512, patch_size=16, in_chans=1,
                 depth=24, num_heads=16, mlp_ratio=2., norm_layer=nn.LayerNorm):
        '''
        Args: 
        -----
            hid_feats_node: int, 
                            the hidden features of NeuralODE model
            hid_feats_decoder: int,
                            the hidden features of decoder model
            n_hid_layers_node: int,
                            the number of hidden layers used for NeuralODE model
        '''


        super(encoder_node_decoder, self).__init__()

        self.hid_feats_node = hid_feats_node
        # self.hid_feats_decoder = hid_feats_decoder
        self.n_hid_layers_node = n_hid_layers_node
        # self.n_res_blocks_decoder = n_res_blocks_decoder
        self.epsilon_node = epsilon_node
        self.substeps_node = substeps_node
        self.h, self.w = high_res
        self.n_sensors = n_sensors
        self.up_factor = up_factor
        self.window_size = window_size
        self.hid_feats_trans = hid_feats_trans
        self.encoder_embed_dim = encoder_embed_dim

        # Neural ODE model, s_t -> s_{t+1}, s variables shape: [b, c=1, n_sensors]
        if tnode == 'node':
            self.node = node(in_feats=self.encoder_embed_dim, hid_feats=self.hid_feats_node, n_hid_layers=self.n_hid_layers_node, epsilon=self.epsilon_node, substeps=self.substeps_node, t_activation=tactivation)
        elif tnode == 'gnode2':
            self.node = gnode2(in_feats=self.encoder_embed_dim, hid_feats=self.hid_feats_node, n_hid_layers=self.n_hid_layers_node, epsilon=self.epsilon_node, substeps=self.substeps_node, t_activation=tactivation)
        
        # transition model, s_{t+1} -> z_{t+1}, z variable shape: [b, 1, (h/r) * (w/r)]
        self.trans_h, self.trans_w = int(self.h / self.up_factor), int(self.w / self.up_factor)
        out_feats = self.trans_h * self.trans_w
        self.transition = trans_net(in_feats=self.encoder_embed_dim*65, hid_feats=[self.hid_feats_trans,self.hid_feats_trans], out_feats=out_feats) 

        # decoder model
        self.decoder = SwinIR(img_size=(self.trans_h, self.trans_w), upscale=self.up_factor, in_chans=1, window_size=self.window_size,
                                img_range=1., mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', embed_dim=60,
                                depths= [6, 6, 6, 6],  
                                num_heads= [6, 6, 6, 6])
        assert self.h == self.w
        self.patch_embed = PatchEmbed(self.w, patch_size, in_chans, self.encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.encoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(self.encoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(self.encoder_embed_dim)

        self.init_mask = [0]*self.n_sensors+[1]*(self.h*self.w - self.n_sensors)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
        

    def forward(self, X, hi_res_result=False):
        '''
        Args:
        -----
        s: torch tensor, shape: [b, c=1, n_sensors] 
           sparse sensor measurement data
        epsilon: float, a hyper-parameter for tuning NODE model
        '''
        B, C, H, W = X.shape
        init_mask = copy.deepcopy(self.init_mask)
        np.random.shuffle(init_mask)
        mask = torch.from_numpy(np.repeat(np.asarray(init_mask).reshape(1, 1, H, W), B, axis=0)).cuda()
        if hi_res_result:
            s = self.forward_encoder(X)
        else:
            s = self.forward_encoder(X*(1-mask))
        # NODE: s_t -> s_{t+1}
        s_next = self.node(s)

        x_next = None

        # Transition: s_{t+1}, [b, c=1, n_sensors] -> z_{t+1}, [b, 1, (h/r) * (w/r)]
        x_next = self.transition(s_next.reshape(B, 1, -1))
        # Transition: reshape to [b, 1, (h/r), (w/r)]
        x_next = x_next.view(x_next.shape[0], x_next.shape[1], int(self.h / self.up_factor), int(self.w / self.up_factor))

        # Decoder: z_{t+1}, [b, 1, (h/r) * (w/r)] -> x_{t+1}, [b, 1, h, w]
        x_next = self.decoder(x_next)
        return x_next, mask #, ids_restore 
    

class fno_baseline(nn.Module):
    '''
    Combine NeuralODE and decoder to predict high-resolution dynamics
        CNODE: learn dynamics on sparse sensor measurements, s_t -> s_{t+1}
        Decoder: upsample to high-resolution dynamics, s_{t+1} -> x_{t+1}
    '''
    def __init__(self, hid_feats_node, n_hid_layers_node, 
                 hid_feats_trans, n_sensors, high_res, up_factor=8):
        '''
        Args: 
        -----
            hid_feats_node: int, 
                            the hidden features of NeuralODE model
            hid_feats_decoder: int,
                            the hidden features of decoder model
            n_hid_layers_node: int,
                            the number of hidden layers used for NeuralODE model
        '''


        super(fno_baseline, self).__init__()

        self.hid_feats_node = hid_feats_node
        self.n_hid_layers_node = n_hid_layers_node
        self.h, self.w = high_res
        self.n_sensors = n_sensors
        self.up_factor = up_factor
        self.hid_feats_trans = hid_feats_trans

        # Neural ODE model, s_t -> s_{t+1}, s variables shape: [b, c=1, n_sensors]
        self.fno = fno()
        # transition model, s_{t+1} -> z_{t+1}, z variable shape: [b, 1, (h/r) * (w/r)]
        self.trans_h, self.trans_w = int(self.h / self.up_factor), int(self.w / self.up_factor)
        out_feats = self.trans_h * self.trans_w
        self.transition = trans_net(in_feats=self.n_sensors, hid_feats=[self.hid_feats_trans,self.hid_feats_trans], out_feats=out_feats) 

        self.resize = v2.Resize((self.h, self.w), interpolation=v2.InterpolationMode.BICUBIC)
        

    def forward(self, s, hi_res_result=False):
        '''
        Args:
        -----
        s: torch tensor, shape: [b, c=1, n_sensors] 
           sparse sensor measurement data
        epsilon: float, a hyper-parameter for tuning NODE model
        '''
        if hi_res_result:
            x = self.transition(s)
            x = x.view(x.shape[0], x.shape[1], int(self.h / self.up_factor), int(self.w / self.up_factor))
            x = self.resize(x)
        else:
            x = s
        x_next = self.fno(x)
        return x_next 
