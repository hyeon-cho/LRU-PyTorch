import torch 
import torch.nn as nn

import math

class Torch_LRU(nn.Module):
    def __init__(self, 
                 in_dim:    int=256, 
                 state_dim: int=512, 
                 out_dim:   int=256, 
                 radius:    dict={'min': .9, 'max': .999},
                 phase_const: float=2. 
                 ): 
        super(Torch_LRU, self).__init__()

        u1 = torch.rand(state_dim)
        u2 = torch.rand(state_dim)

        # radius of eigenvalues
        r_min, r_max = radius['min'], radius['max']

        # phase 
        _phase_scaler = phase_const
        self.phase    = math.pi * _phase_scaler
        self.nu_log   = nn.Parameter(torch.log(-0.5*torch.log(u1*(r_max+r_min)*(r_max-r_min) + r_min**2)))
        self.theta_log = nn.Parameter(torch.log(self.phase*u2))

        # magnitude 
        _magnitude     = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(_magnitude) - torch.square(_magnitude))))

        # weights
        B_real = torch.randn(state_dim, in_dim)  / math.sqrt(2*in_dim)
        B_imag = torch.randn(state_dim, in_dim)  / math.sqrt(2*in_dim)
        C_real = torch.randn(out_dim, state_dim) / math.sqrt(state_dim)
        C_imag = torch.randn(out_dim, state_dim) / math.sqrt(state_dim)

        self.B = nn.Parameter(torch.complex(B_real, B_imag))
        self.C = nn.Parameter(torch.complex(C_real, C_imag))

        self.D = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))

    def _cast_to_complex(self, u):
        if u.dtype.is_floating_point:
            u = u.to(dtype=torch.complex64)
        return u

    # jax: Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
    def _calcu_lambda(self):
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re  = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im  = Lambda_mod * torch.sin(torch.exp(self.theta_log))

        return torch.complex(Lambda_re, Lambda_im).to(self.B.device)

    # jax: Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
    def _get_lambda_elems(self): 
        return self._calcu_lambda().unsqueeze(0)

    # jax: B_norm = (B_re + 1j*B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    def _calcu_B_norm(self): 
        B_re = self.B.real
        B_im = self.B.imag
        return (B_re + 1j*B_im) * torch.exp(self.gamma_log).unsqueeze(-1)

    # jax: Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    def _get_Bu_elems(self, u):
        u      = u.squeeze(0)
        B_norm = self._calcu_B_norm()
        return torch.einsum('ij,bj->bi', B_norm, self._cast_to_complex(u)).unsqueeze(0)

    def feedforward_scan(self, Lambda_elements, Bu_elements, x0):
        assert x0 is not None , "x0 must be provided"
        return Lambda_elements[0] * x0 + Bu_elements

    def forward(self, u, x0=None): 
        state = self.feedforward_scan(
            self._get_lambda_elems(), 
            self._get_Bu_elems(u), 
            x0=x0
        )

        # y = (self.C @ state).real + self.D @ u
        y   =     torch.einsum("bnh,oh->bno", state, self.C).real
        y   = y + torch.einsum("bnl,ol->bno", u, self.D)
        return y, state
