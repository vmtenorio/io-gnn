import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pytorch_lightning as pl
import math

class GCNLayer(nn.Module):
    def __init__(self, S, in_dim, out_dim):
        super().__init__()

        self.S = S
        self.N = self.S.shape[0]
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W1 = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W1.data)

        self.bias = nn.Parameter(torch.empty(self.out_dim))
        std = math.sqrt(self.in_dim)
        torch.nn.init.uniform_(self.bias.data, -std, std)

        self.S = nn.Parameter(self.S, requires_grad=False)

    def forward(self, x):
        if x.ndim == 3:
            _, xN, xFin = x.shape
        assert xN == self.N
        assert xFin == self.in_dim

        return self.S @ x @ self.W1 + self.bias[None,None,:]

class GCNHLayer(nn.Module):
    def __init__(self, Spow, in_dim, out_dim):
        super().__init__()
        self.K = Spow.shape[0]
        self.N = Spow.shape[1]

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.empty((self.K, self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        self.bias = nn.Parameter(torch.empty(self.out_dim))
        std = math.sqrt(self.in_dim)
        torch.nn.init.uniform_(self.bias.data, -std, std)

        self.Spow = nn.Parameter(Spow, requires_grad=False)

    def forward(self, x):
        # x is of shape T x N x F
        assert x.ndim == 3
        T, xN, xFin = x.shape
        assert xN == self.N
        assert xFin == self.in_dim
        #x_rep = x.repeat([self.K, 1, 1, 1])
        #return (self.Spow[:,None,:,:] @ x_rep @ self.W[:,None,:,:]).sum(0)
        result = torch.zeros((T, self.N, self.out_dim), device=x.device)
        for k in range(self.K):
            result += self.Spow[k,:,:] @ x @ self.W[k,:,:]

        return result + self.bias[None,None,:]


class GCN(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, nonlin=torch.nn.Tanh):
        super().__init__()

        self.N = S.shape[0]
        d = S.sum(1)
        d = torch.where(d != 0, d, 1)
        if not torch.all(d > 0):
            print("Isolated node (degree 0)")
        d_norm = torch.sqrt(1/d)
        D_sqrt = torch.diag(d_norm)
        S_i = S + torch.eye(self.N, device=S.device)

        self.S = D_sqrt @ S_i @ D_sqrt

        self.n_layers = n_layers
        self.nonlin = nonlin()
        self.convs = nn.ModuleList()

        if n_layers > 1:
            self.convs.append(GCNLayer(self.S, in_dim, hid_dim))
            for _ in range(n_layers - 2):
                self.convs.append(GCNLayer(self.S, hid_dim, hid_dim))
            self.convs.append(GCNLayer(self.S, hid_dim, out_dim))
        else:
            self.convs.append(GCNLayer(self.S, in_dim, out_dim))


    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
        x = self.convs[-1](x)

        return x
    
class GCNH(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K, norm_S="degree", nonlin=torch.nn.Tanh):
        super().__init__()

        if norm_S == "degree":

            self.N = S.shape[0]
            d = S.sum(1)
            d = torch.where(d != 0, d, 1)
            if not torch.all(d > 0):
                print("Isolated node (degree 0)")
            d_norm = torch.sqrt(1/d)
            D_sqrt = torch.diag(d_norm)
            S_i = S + torch.eye(self.N, device=S.device)

            self.S = D_sqrt @ S_i @ D_sqrt

        elif norm_S == "eigval":
            vals = torch.linalg.eigvalsh(S)
            self.S = S / vals.max()
        else:
            self.S = S

        self.K = K
        self.Spow = torch.stack([torch.linalg.matrix_power(self.S, k) for k in range(self.K)], dim=0)

        self.n_layers = n_layers
        self.nonlin = nonlin()
        self.convs = nn.ModuleList()

        if n_layers > 1:
            self.convs.append(GCNHLayer(self.Spow, in_dim, hid_dim))
            for _ in range(n_layers - 2):
                self.convs.append(GCNHLayer(self.Spow, hid_dim, hid_dim))
            self.convs.append(GCNHLayer(self.Spow, hid_dim, out_dim))
        else:
            self.convs.append(GCNHLayer(self.Spow, in_dim, out_dim))


    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
        x = self.convs[-1](x)

        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, nonlin=torch.nn.Tanh):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.nonlin = nonlin()

        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(nn.Linear(in_dim, hid_dim))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.layers[i](x))
        x = self.layers[-1](x)

        return x

class IOArch(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, idxs, method, nonlin=torch.nn.Tanh, C=None):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.nonlin = nonlin
        self.Nin = Nin
        self.Nout = Nout
        self.idxs = idxs
        assert method in ["fill", "linear", "transpose", "nothing", "selection"], "Not a valid method chosen"
        self.method = method
        
        if self.method == "transpose":
            self.X_out_dim = self.Nout
            self.Y_in_dim = self.Nin
        else:
            self.X_out_dim = self.hid_dim
            self.Y_in_dim = self.hid_dim

        if self.method == "linear":
            self.linear_transform = nn.Parameter(torch.empty((self.Nout, self.Nin)))
            torch.nn.init.kaiming_uniform_(self.linear_transform.data)

        if self.method == "selection":
            assert C is not None
            self.C = C

    def forward(self, gx, gy, feat):
        #assert feat.shape[0] == self.Nin
        Z = self.arch_X(feat)
        if self.method == "fill":
            Zy = torch.ones((self.Nout, Z.shape[1]), device=feat.device)
            Zy[self.idxs] = Z
        elif self.method == "linear":
            Zy = self.linear_transform @ Z
        elif self.method == "transpose":
            if Z.ndim > 2:
                Zy = Z.permute(0,2,1)
            else:
                Zy = Z.T
        elif self.method == "nothing":
            Zy = Z
        elif self.method == "selection":
            Zy = self.C[None,:,:] @ Z

        return self.arch_Y(Zy)
    
    def get_embeddings(self, feat):
        return self.arch_X(feat)

class IOGCN(IOArch):
    def __init__(self, Sin, Sout, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, nonlin=torch.nn.Tanh, C=None):
        super().__init__(in_dim, hid_dim, out_dim, Nin, Nout, idxs, method, nonlin=nonlin, C=C)

        self.arch_X = GCN(Sin, in_dim, hid_dim, self.X_out_dim, n_layers, nonlin=self.nonlin)
        self.arch_Y = GCN(Sout, self.Y_in_dim, hid_dim, out_dim, n_layers, nonlin=self.nonlin)

class IOGCNH(IOArch):
    def __init__(self, Sin, Sout, in_dim, hid_dim, out_dim, Nin, Nout, n_layers_x, n_layers_y, K, idxs, method, nonlin=torch.nn.Tanh, C=None, norm_S="degree"):
        super().__init__(in_dim, hid_dim, out_dim, Nin, Nout, idxs, method, nonlin=nonlin, C=C)

        self.arch_X = GCNH(Sin, in_dim, hid_dim, self.X_out_dim, n_layers_x, K=K, nonlin=self.nonlin, norm_S=norm_S)
        self.arch_Y = GCNH(Sout, self.Y_in_dim, hid_dim, out_dim, n_layers_y, K=K, nonlin=self.nonlin, norm_S=norm_S)

class IOMLP(IOArch):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, nonlin=torch.nn.Tanh, C=None):
        super().__init__(in_dim, hid_dim, out_dim, Nin, Nout, idxs, method, nonlin=nonlin, C=C)

        self.arch_X = MLP(in_dim, hid_dim, self.X_out_dim, n_layers, nonlin=self.nonlin)
        self.arch_Y = MLP(self.Y_in_dim, hid_dim, out_dim, n_layers, nonlin=self.nonlin)

