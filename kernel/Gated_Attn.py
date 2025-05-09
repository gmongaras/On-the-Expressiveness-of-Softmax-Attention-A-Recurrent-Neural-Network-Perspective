import torch
import math


def clamp_exp_der(X, max_val):
    return X.clamp(max=max_val).exp() * (X <= max_val)


class GatedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, G_O, G_I, M, attn_scale):
        if type(attn_scale) == float:
            attn_scale = torch.tensor(attn_scale, requires_grad=False)
        ctx.save_for_backward(Q, K, V, G_O, G_I, M, attn_scale)
        return G_O * ((((Q @ K.mT) * attn_scale).clamp(max=5).exp() * M) @ (V * G_I))

    @staticmethod
    def backward(ctx, prev_grad):
        Q, K, V, G_O, G_I, M, attn_scale = ctx.saved_tensors

        dtype = torch.float64 if prev_grad.dtype == torch.float64 else torch.float32
        Q = Q.to(dtype)
        K = K.to(dtype)
        V = V.to(dtype)
        G_O = G_O.to(dtype)
        G_I = G_I.to(dtype)
        M = M.to(dtype)

        A_pre = (Q @ K.mT) * attn_scale
        A_prime = clamp_exp_der(A_pre, 5)
        A = A_pre.clamp(max=5).exp()
        A_M = A * M
        V_G = V * G_I
        A_bar = A_M @ V_G
        D_G = G_O * prev_grad

        A_M_D_G = (A_M.mT @ D_G)
        Aprime_D_V_M = (A_prime * ((D_G @ V_G.mT) * M))
        
        G_O_grad = prev_grad * A_bar
        G_I_grad = V * A_M_D_G
        V_grad = G_I * A_M_D_G
        Q_grad = Aprime_D_V_M @ K * attn_scale
        K_grad = Aprime_D_V_M.mT @ Q * attn_scale
        
        return Q_grad, K_grad, V_grad, G_O_grad, G_I_grad, None, None
    
    
if __name__ == "__main__":
    N = 100
    d = 5
    e = 4
    M = 110
    Q = torch.randn(N, d, requires_grad=True) * 10
    K = torch.randn(M, d, requires_grad=True)
    V = torch.randn(M, e, requires_grad=True)
    M_ = torch.tril(torch.ones(N, M)).bool().requires_grad_(False)
    G_O = torch.randn(N, 1, requires_grad=True)
    G_I = torch.randn(M, 1, requires_grad=True)
    attn_scale = torch.tensor(math.sqrt(d))


    torch.autograd.gradcheck(GatedAttention.apply, (Q.double(), K.double(), V.double(), G_O.double(), G_I.double(), M_.bool(), attn_scale), eps=1e-4)