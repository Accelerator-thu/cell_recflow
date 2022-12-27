import torch
import torch.nn as nn
import numpy as np
from tqdm import trange


class MLP(nn.Module):
    def __init__(self, input_dim=400, hidden_num=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    
    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


class RectifiedFlow():
    def __init__(self, 
                #  index_pair_list, 
                 n_obs_list, 
                 model=None, 
                 num_steps=1000, 
                 reverse = False
                #  adata_active=True
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.index_pair_list = [(i, i + 1) for i in range(38)]
        self.model = model
        self.N = num_steps
        self.reverse = reverse
        # self.adata_active = adata_active
        self.crt_pcadata = {}
        self.ref_crt_pcadata = {}
        self.new_crt_pcadata = {}
        for index_pair in self.index_pair_list:
            self.crt_pcadata[index_pair] = torch.from_numpy(np.load(f'pcadata/pcadata_contiguous_{index_pair[0]}_{index_pair[1]}.npy', allow_pickle=True)[0]).to(self.device)
            self.ref_crt_pcadata[index_pair[0]] = self.crt_pcadata[index_pair][:n_obs_list[index_pair[0]]]
            self.new_crt_pcadata[index_pair[1]] = self.crt_pcadata[index_pair][n_obs_list[index_pair[0]]:]
        # print(f"Model on {self.device}")
        self.model.to(self.device)
        # self.crt_pcadata.to(self.device)
        
        # self.ref_crt_pcadata = self.crt_pcadata[:n_obs_list[index_pair[0]]]
        # self.new_crt_pcadata = self.crt_pcadata[n_obs_list[index_pair[0]]:]
    
    def get_train_tuple(self, z0=None, z1=None, idp=None):
        t = torch.rand((z1.shape[0], 1)).to(self.device)
        # print(t.shape)
        # print(z1)
        # print(z1.shape)
        # if self.adata_active:
        tmpz1 = torch.unsqueeze(torch.Tensor(self.new_crt_pcadata[idp[0, 1].item()][z1[0]]), 0)
        tmpz0 = torch.unsqueeze(torch.Tensor(self.ref_crt_pcadata[idp[0, 0].item()][z0[0]]), 0)
        for i in range(1, len(z0)):
            tmpz1 = torch.concat((tmpz1, torch.unsqueeze(torch.Tensor(self.new_crt_pcadata[idp[i, 1].item()][z1[i]]), 0)), 0)
            tmpz0 = torch.concat((tmpz0, torch.unsqueeze(torch.Tensor(self.ref_crt_pcadata[idp[i, 0].item()][z0[i]]), 0)), 0)
        if not self.reverse:
            z1, z0 = tmpz1, tmpz0
        else:
            z1, z0 = tmpz0, tmpz1
        # print(z1)
        # print(z1.shape)
        
        z_t =  t * z1 + (1. - t) * z0
        target = z1 - z0 
            
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, zid=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N    
        dt = 1. / N
        traj = [] # to store the trajectory
        # print(z0)
        z = torch.Tensor(self.ref_crt_pcadata[zid][z0]).detach().clone()
        # print(z)
        batchsize = z.shape[0]
        # z.shape
        traj.append(z.detach().clone())
        z = z.to(self.device)
        for i in range(N):
            t = torch.ones((batchsize,1)) * i / N
            # print(z)
            t = t.to(self.device)
            # print("t: ", t.device)
            # print("z: ", z.device)
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
            z = z.to(self.device)
            

        return traj
    
    @torch.no_grad()
    def simulate_target(self, z0=None, zid=None, N=None):
        if N is None:
            N = self.N    
        dt = torch.tensor(1. / N).to(self.device)
        if not self.reverse:
            z = torch.Tensor(self.ref_crt_pcadata[zid][z0]).detach().clone()
        else:
            z = torch.Tensor(self.new_crt_pcadata[zid][z0]).detach().clone()
            
        # print(z)
        batchsize = z.shape[0]
        # z.shape
        z = z.to(self.device)
        for i in range(N):
            t = torch.ones((batchsize,1)) * i / N
            # print(z)
            t = t.to(self.device)
            # print("t: ", t.device)
            # print("z: ", z.device)
            pred = self.model(z, t)
            z = z + pred * dt
        return z
    

def train_rectified_flow(rectified_flow, optimizer, idpairs, batchsize, inner_iters):
    loss_curve = []
    for i in trange(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(idpairs))[:batchsize]
        batch = idpairs[indices]
        # print(batch)
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        idp = batch[:, 2:4].detach().clone()
        # print(z0.max())
        # print(z1.max())
        # print(pcadatalist[1][z1])
        
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1, idp=idp)
        z_t = z_t.to(rectified_flow.device)
        t = t.to(rectified_flow.device)
        target = target.to(rectified_flow.device)

        pred = rectified_flow.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        loss_curve.append(np.log(loss.item())) ## to store the loss curve

    return rectified_flow, loss_curve