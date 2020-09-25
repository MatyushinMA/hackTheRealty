import torch
import torch.nn as nn
from fastFM.mcmc import FMRegression
from scipy.sparse import csc_matrix

def dense_to_sparse(dense):
    sparse = csc_matrix(dense.shape)
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            sparse[i, j] = dense[i, j]
    return sparse

class DSMM(nn.Module):
    def __init__(self, in_size, out_size, depth=3, activation=nn.Tanh):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layers = []
        for i in range(self.depth):
            i_size = int(in_size*(1 - (i/depth)) + (out_size*i/depth))
            o_size = int(in_size*(1 - ((i + 1)/depth)) + (out_size*(i + 1)/depth))
            lin_layer = nn.Linear(i_size, o_size)
            act_layer = nn.PReLU(1)
            self.layers.extend([lin_layer, act_layer])
        self.layers.append(activation())

    def nop(self):
        self.numel = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return self.numel

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class NNBlendFM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size,
                       nn_depth=3,
                       nn_activation=nn.Tanh,
                       fm_rank=2,
                       fm_warm_up=1000,
                       fm_iter=100):
        super().__init__()
        self.embedder = DSSM(in_size, hidden_size, depth=nn_depth, activation=nn_activation)
        self.fm = [FMRegression(n_iter=fm_warm_up, rank=fm_rank) for _ in range(out_size)]
        self.fm_iter = fm_iter
        self.cold = True

    def machine_forward(self, x):
        def single_machine_forward(fm):
            selfdot = x.unsqueeze(-1)*x.unsqueeze(1)
            vdot = torch.from_numpy(np.dot(fm.V_.T, fm.V_))
            return fm.w0_ + torch.matmul(x, torch.from_numpy(fm.w_).float()) + torch.sum(selfdot*(1 - torch.eye(5))*vdot, dim=(1, 2))/2
        return torch.stack(map(single_machine_forward, self.fm))

    def forward(self, x):
        x = self.embedder(x)
        return self.machine_forward(x)

    def fit_machine(self, x, y, x_val, y_val):
        mse = 0
        embeds = self.embedder(torch.from_numpy(x).float())
        val_embeds = self.embedder(torch.from_numpy(x_val).float())
        embeds = dense_to_sparse(embeds)
        val_embeds = dense_to_sparse(val_embeds)
        if self.cold:
            for i, fm in enumerate(self.fm):
                hat = fm.fit_predict(embeds, y[:, i], val_embeds)
                mse += np.sum((hat - y_val[:, i])**2)
            self.cold = False
        else:
            for i, fm in enumerate(self.fm):
                hat = fm.fit_predict(embeds, y[:, i], val_embeds, n_more_iter=self.fm_iter)
                mse += np.sum((hat - y_val[:, i])**2)
        return mse
