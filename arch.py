import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pytorch_lightning as pl


class GNNBase(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, nonlin, build_params):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.nonlin_f = nonlin()

        self.build(build_params)

    def build(self, build_params):
        # To be implemented in child classes
        pass


    def forward(self, graph, x):
        
        if x.ndim == 3:
            x = x.permute(1,0,2)

        for i in range(self.n_layers - 1):
            x = self.nonlin_f(self.convs[i](graph, x)).squeeze()
        x = self.convs[-1](graph, x).squeeze()

        if x.ndim == 3:
            return x.permute(1,0,2)
        else:
            return x
    
class GCN(GNNBase):
    def build(self, build_params):
        if self.n_layers > 1:
            self.convs.append(dgl.nn.GraphConv(self.in_dim, self.hid_dim, norm=build_params['norm'], bias=build_params.get('bias', True)))
            for i in range(self.n_layers - 2):
                self.convs.append(dgl.nn.GraphConv(self.hid_dim, self.hid_dim, norm=build_params['norm'], bias=build_params.get('bias', True)))
            self.convs.append(dgl.nn.GraphConv(self.hid_dim, self.out_dim, norm=build_params['norm'], bias=build_params.get('bias', True)))
        else:
            self.convs.append(dgl.nn.GraphConv(self.in_dim, self.out_dim, norm=build_params['norm'], bias=build_params.get('bias', True)))
    
class GAT(GNNBase):
    def build(self, build_params):
        if self.n_layers > 1:
            self.convs.append(dgl.nn.GATConv(self.in_dim, self.hid_dim, build_params['num_heads'], feat_drop=build_params.get('feat_drop', 0.), attn_drop=build_params.get('attn_drop', 0.)))
            for i in range(self.n_layers - 2):
                self.convs.append(dgl.nn.GATConv(self.hid_dim*build_params['num_heads'], self.hid_dim, build_params['num_heads'], feat_drop=build_params.get('feat_drop', 0.), attn_drop=build_params.get('attn_drop', 0.)))
            self.convs.append(dgl.nn.GATConv(self.hid_dim*build_params['num_heads'], self.out_dim, 1, feat_drop=build_params.get('feat_drop', 0.), attn_drop=build_params.get('attn_drop', 0.)))
        else:
            self.convs.append(dgl.nn.GATConv(self.in_dim, self.out_dim, 1, feat_drop=build_params.get('feat_drop', 0.), attn_drop=build_params.get('attn_drop', 0.)))
    
    def forward(self, graph, x):
        if x.ndim == 3:
            x = x.permute(1,0,2)
            x_shape = (x.shape[0], x.shape[1], -1)
        else:
            x_shape = (x.shape[0], -1)

        for i in range(self.n_layers - 1):
            x = self.nonlin_f(self.convs[i](graph, x)).reshape(x_shape)
        x = self.convs[-1](graph, x).reshape(x_shape)

        if x.ndim == 3:
            return x.permute(1,0,2)
        else:
            return x


class GCNHLayer(nn.Module):
    def __init__(self, S, in_dim, out_dim, K):
        super().__init__()

        self.S = S
        self.N = self.S.shape[0]
        self.S += torch.eye(self.N, device=self.S.device)
        self.d = self.S.sum(1)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        self.S = self.D_inv @ self.S @ self.D_inv

        self.K = K
        self.Spow = torch.zeros((self.K, self.N, self.N), device=self.S.device)
        self.Spow[0,:,:] = torch.eye(self.N, device=self.S.device)
        for k in range(1, self.K):
            self.Spow[k,:,:] = self.Spow[k-1,:,:] @ self.S

        self.Spow = nn.Parameter(self.Spow, requires_grad=False)

        self.S = nn.Parameter(self.S, requires_grad=False)

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty(self.K, self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data)

    def forward(self, x):
        assert (self.N, self.in_dim) == x.shape
        out = torch.zeros((self.N, self.out_dim), device=x.device)
        for k in range(self.K):
            out += self.Spow[k,:,:] @ x @ self.W[k,:,:]
        return out

class GCNH(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, K, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        if n_layers > 1:
            self.convs.append(GCNHLayer(S, in_dim, hid_dim, K))
            for i in range(n_layers - 2):
                self.convs.append(GCNHLayer(S, hid_dim, hid_dim, K))
            self.convs.append(GCNHLayer(S, hid_dim, out_dim, K))
        else:
            self.convs.append(GCNHLayer(S, in_dim, out_dim, K))


    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = torch.tanh(self.convs[i](x))
        x = self.convs[-1](x)

        return x
    

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, nonlin):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.nonlin_f = nonlin()

        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(nn.Linear(in_dim, hid_dim))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, graph, x): # Graph kept for compatibility with GCN

        for i in range(self.n_layers - 1):
            x = self.nonlin_f(self.layers[i](x))
        x = self.layers[-1](x)

        return x

class IOArch(nn.Module):
    def __init__(self, Nin, Nout, hid_dim, idxs, method, C=None): # TODO set idxs as an optional argument
        super().__init__()
        self.Nin = Nin
        self.Nout = Nout
        self.idxs = idxs
        assert method in ["fill", "linear", "graph", "transpose", "nothing", "selection", "common"], "Not a valid method chosen"
        self.method = method
        
        if self.method == "transpose":
            self.X_out_dim = self.Nout
            self.Y_in_dim = self.Nin

        else:
            self.X_out_dim = hid_dim
            self.Y_in_dim = hid_dim

        if self.method == "linear":
            self.linear_transform = nn.Parameter(torch.empty((self.Nout, self.Nin)))
            torch.nn.init.kaiming_uniform_(self.linear_transform.data)
        
        if self.method == "selection":
            assert C is not None
            self.C = C

    def forward(self, graph1, graph2, feat):
        #assert feat.shape[0] == self.Nin
        if self.arch_X.n_layers > 0:
            Z = self.arch_X(graph1, feat)
        else:
            Z = feat
            
        if self.method == "fill":
            Zy = torch.ones((self.Nout, Z.shape[1]), device=feat.device)
            Zy[self.idxs] = Z
        elif self.method == "linear":
            Zy = self.linear_transform @ Z
        elif self.method == "graph":
            adj = graph2.adjacency_matrix().to_dense().to(feat.device)
            Zy = adj[:,self.idxs] @ Z
        elif self.method == "transpose":
            if Z.ndim > 2:
                Zy = Z.permute(0,2,1)
            else:
                Zy = Z.T
        elif self.method == "nothing":
            Zy = Z
        elif self.method == "selection":
            if Z.ndim == 3:
                Zy = self.C[None,:,:] @ Z
            else:
                Zy = self.C @ Z
        elif self.method == "common":
            Zy = torch.zeros((self.Nout, Z.shape[1]), device=feat.device)
            Zy[self.idxs[1],:] = Z[self.idxs[0],:]
        return self.arch_Y(graph2, Zy)
    
    def get_embeddings(self, graph1, feat):
        return self.arch_X(graph1, feat)

class IOGCN(IOArch):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, build_params, n_layers_y=-1, nonlin=torch.nn.Tanh, C=None):
        super().__init__(Nin, Nout, hid_dim, idxs, method, C=C)

        if n_layers_y < 0:
            n_layers_y = n_layers
        if n_layers == 0: # Only output GCN after transformation between graphs
            self.Y_in_dim = in_dim

        self.arch_X = GCN(in_dim, hid_dim, self.X_out_dim, n_layers, nonlin, build_params)
        self.arch_Y = GCN(self.Y_in_dim, hid_dim, out_dim, n_layers_y, nonlin, build_params)

class IOGCNH(IOArch):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, build_params, n_layers_y=-1, nonlin=torch.nn.Tanh, C=None):
        super().__init__(Nin, Nout, hid_dim, idxs, method, C=C)

        if n_layers_y < 0:
            n_layers_y = n_layers
        if n_layers == 0: # Only output GCN after transformation between graphs
            self.Y_in_dim = in_dim

        self.arch_X = GCNH(build_params['Sx'], in_dim, hid_dim, self.X_out_dim, build_params['K'], n_layers)
        self.arch_Y = GCNH(build_params['Sy'], self.Y_in_dim, hid_dim, out_dim, build_params['K'], n_layers_y)

class IOGAT(IOArch):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, build_params, n_layers_y=-1, nonlin=torch.nn.Tanh, C=None):
        super().__init__(Nin, Nout, hid_dim, idxs, method, C=C)

        if n_layers_y < 0:
            n_layers_y = n_layers
        if n_layers == 0: # Only output GAT after transformation between graphs
            self.Y_in_dim = in_dim

        self.arch_X = GAT(in_dim, hid_dim, self.X_out_dim, n_layers, nonlin, build_params)
        self.arch_Y = GAT(self.Y_in_dim, hid_dim, out_dim, n_layers_y, nonlin, build_params)


class IOMLP(IOArch):
    def __init__(self, in_dim, hid_dim, out_dim, Nin, Nout, n_layers, idxs, method, build_params, n_layers_y=-1, nonlin=torch.nn.Tanh, C=None):
        super().__init__(Nin, Nout, hid_dim, idxs, method, C=C)

        if n_layers_y < 0:
            n_layers_y = n_layers

        self.arch_X = MLP(in_dim, hid_dim, self.X_out_dim, n_layers, nonlin)
        self.arch_Y = MLP(self.Y_in_dim, hid_dim, out_dim, n_layers_y, nonlin)
