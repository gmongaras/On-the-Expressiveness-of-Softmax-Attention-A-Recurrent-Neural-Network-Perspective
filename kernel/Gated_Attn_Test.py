import torch


class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, M, G_O, G_I):
        ctx.save_for_backward(Q, K, V, M, G_O, G_I)
        return G_O * (((Q @ K.mT).exp() * M) @ (V * G_I))

    @staticmethod
    def backward(ctx, prev_grad):
        Q, K, V, M, G_O, G_I = ctx.saved_tensors

        A = (Q @ K.mT).exp()
        A_M = A * M
        V_G = V * G_I
        A_bar = A_M @ V_G
        D_G = G_O * prev_grad
        
        G_O_grad = prev_grad * A_bar
        G_I_grad = V * (A_M.mT @ D_G)
        Q_grad = (A * ((D_G @ V_G.mT) * M)) @ K
        K_grad = (A.mT * ((V_G @ D_G.mT) * M.mT)) @ Q
        V_grad = (A_M.mT @ D_G) * G_I
        
        return Q_grad, K_grad, V_grad, None, G_O_grad, G_I_grad
    
    
N = 10
d = 5
e = 4
M = 11
Q = torch.rand(N, d, requires_grad=True)
K = torch.rand(M, d, requires_grad=True)
V = torch.rand(M, e, requires_grad=True)
M_ = torch.tril(torch.ones(N, M)).float().requires_grad_(False)
G_O = torch.rand(N, 1, requires_grad=True)
G_I = torch.rand(M, 1, requires_grad=True)


def test():
    torch.autograd.gradcheck(Function.apply, (Q.double(), K.double(), V.double(), M_.double(), G_O.double(), G_I.double()), eps=1e-4)

test()