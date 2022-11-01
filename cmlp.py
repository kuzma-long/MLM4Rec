import copy

import torch
import torch.nn.functional as F
from torch import nn, einsum



class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class ConvGatingUnit(nn.Module):
    def __init__(self, eunits, act=nn.GELU(), dropout=nn.Identity(), cnn_kernel=15, cmlp_type=1,bias=True):
        super().__init__()
        dim_out = eunits // 2
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = dropout
        self.cmlp=cmlp_type
        self.conv1 = nn.Conv1d(
            dim_out,
            dim_out,
            3,
            stride=1,
            padding=1,
            groups=dim_out,
            bias=bias,
        )
        self.conv2 = nn.Conv1d(
            dim_out,
            dim_out,
            5,
            stride=1,
            padding=2,
            groups=dim_out,
            bias=bias,
        )
        if cmlp_type==2:
            self.conv3 = nn.Conv1d(
                dim_out,
                dim_out,
                7,
                stride=1,
                padding=3,
                groups=dim_out,
                bias=bias,
            )
        if cmlp_type==3:
            self.conv3 = nn.Conv1d(
                dim_out,
                dim_out,
                7,
                stride=1,
                padding=3,
                groups=dim_out,
                bias=bias,
            )
            self.conv4 = nn.Conv1d(
                dim_out,
                dim_out,
                9,
                stride=1,
                padding=4,
                groups=dim_out,
                bias=bias,
            )
        self.conv = nn.Conv1d(
            dim_out,
            dim_out,
            cnn_kernel,
            stride=1,
            padding=(cnn_kernel-1)//2,
            groups=dim_out,
            bias=bias,
        )
        self.act = act
        self.conv_act = nn.SiLU()

    def forward(self, x, tiny_attn=None):
        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)
        gate = gate.permute(0, 2, 1)
        gate1=self.conv1(gate)
        gate2=self.conv2(gate)
        gate = self.conv_act(gate1 * gate2)
        if self.cmlp==2:
            gate3=self.conv3(gate)
            gate = self.conv_act(gate1 * gate2 * gate3)
        if self.cmlp==3:
            gate3 = self.conv3(gate)
            gate4=self.conv4(gate)
            gate = self.conv_act(gate1*gate2*gate3*gate4)
        # gate = self.conv(gate)
        # gate = self.conv_act(gate)
        gate = self.dropout(gate)
        gate = gate.permute(0, 2, 1)

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class ConvGatingUnit2(nn.Module):
    def __init__(self, eunits, act=nn.GELU(), dropout=nn.Identity(), cnn_kernel=15, bias=True):
        super().__init__()
        dim_out = eunits // 2
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = dropout
        self.depthwise_conv = nn.Conv1d(
            dim_out,
            dim_out,
            cnn_kernel,
            stride=1,
            padding=(cnn_kernel - 1) // 2,
            groups=dim_out,
            bias=bias,
        )

        self.pointwise_conv = nn.Conv1d(
            dim_out,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.act = act
        self.conv_act = nn.SiLU()

    def forward(self, x, tiny_attn=None):
        res, gate = x.chunk(2, dim=-1)

        gate = self.norm(gate)
        gate = gate.permute(0, 2, 1)
        gate = self.depthwise_conv(gate)
        gate = self.conv_act(gate)
        gate = self.dropout(gate)
        gate = self.pointwise_conv(gate)
        gate = gate.permute(0, 2, 1)

        if tiny_attn is not None:
            gate = gate + tiny_attn

        return self.act(gate) * res


class CMLPBlock(nn.Module):
    def __init__(
            self,
            adim,
            eunits,
            act=nn.GELU(),
            act_in=nn.GELU(),
            act_out=nn.Identity(),
            attn_dim=0,
            causal=False,
            cmlp_type=1,
            kernel=15,
            dropout=0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(adim, eunits)
        self.proj_out = nn.Linear(eunits // 2, adim)
        self.act_in = act_in
        self.act_out = act_out
        self.pre_norm = nn.LayerNorm(adim)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.cgu = ConvGatingUnit(eunits, act, self.dropout, cnn_kernel=kernel,cmlp_type=cmlp_type)
        # elif cmlp_type == 2:
        #     self.cgu = ConvGatingUnit2(eunits, act, self.dropout, cnn_kernel=kernel)
        # else:
        #     raise NotImplementedError

        self.attn = Attention(adim, eunits // 2, attn_dim, causal) if attn_dim > 0 else None

    def forward(self, x):
        residual = x

        x = self.pre_norm(x)

        tiny_attn = self.attn(x) if self.attn is not None else None

        x = self.proj_in(x)
        x = self.act_in(x)

        x = self.cgu(x, tiny_attn=tiny_attn)

        x = self.proj_out(x)
        x = self.act_out(x)

        return residual + x


class CMLPEncoder(nn.Module):
    def __init__(
            self,
            args,
            kernel=3,
            attn_dim=0,
            causal=False,
    ):
        super().__init__()

        self.layers = CMLPBlock(
                    adim=args.hidden_size,
                    eunits=args.hidden_size*4,
                    attn_dim=attn_dim,
                    causal=causal,
                    act=nn.GELU(),
                    act_in=nn.GELU(),
                    act_out=nn.Identity(),
                    cmlp_type=args.cmlp_type,
                    kernel=kernel,
                    dropout=args.hidden_dropout_prob,
                )
        self.norm = nn.LayerNorm(args.hidden_size)

    def forward(self, x):
        out = self.layers(x)
        out = self.norm(out)
        return out

