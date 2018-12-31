import math
import torch
from torch import nn
from torch.nn import functional as F

HID_DIM = 768
D = HID_DIM
dropout = 0.1
Nh = 1 #num heads

Dk = D // Nh
Dv = D // Nh

class simpleBiLinear(nn.Module):
    def __init__(self, config):
        super(simpleBiLinear, self).__init__()
        self.config = config
        self.embed_dim = self.config['embed_dim']
        self.hid_dim = self.config['hid_dim']
        self.dropout_rate = self.config['dropout']

        self.linear1 = nn.Linear(self.embed_dim, self.hid_dim, True)
        self.bn1 = nn.BatchNorm1d(self.hid_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.linear2 = nn.Linear(self.hid_dim, self.hid_dim, True)
        self.bn2 = nn.BatchNorm1d(self.hid_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.linear_out = nn.Linear(self.hid_dim, 1, True)

    # title: B x embed_dim
    # tag: B x embed_dim
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)

        outs = self.linear_out(x)
        outs = torch.sigmoid(outs)

        return outs


class attention(nn.Module):
    def __init__(self, config):
        super(attention, self).__init__()
        self.config = config
        self.embed_dim = self.config['embed_dim']
        self.hid_dim = self.config['hid_dim']
        self.dropout = self.config['dropout']

        limit = math.sqrt(1 / self.embed_dim)

        W = torch.empty((3 * self.embed_dim), dtype=torch.float)
        nn.init.uniform_(W, -limit, limit)
        self.W = nn.Parameter(W)

        self.resizer = DepthwiseSeparableConv(4 * D, D, 5)

        self.enc_blk = EncoderBlock(conv_num=2, ch_num=D, k=3, length=self.config['MAX_TITLE_LEN'], hid_dim=D)
        self.model_enc_blks = nn.ModuleList([self.enc_blk] * 2)

        self.fc_final = nn.Linear(config['MAX_TITLE_LEN'] * D, 1)

    # input: [Batch_size, TITLE_LEN + TAG_LEN, EMD_DIM]  eg.[62, 31, 768]
    # C: context, title [Batch_size, TITLE_LEN, EMD_DIM] eg.[62, 27, 768]
    # Q: query, tag [Batch_size, TAG_LEN, EMD_DIM] eg.[62, 4, 768]
    def forward(self, x):
        C = x[:, :self.config['MAX_TITLE_LEN'], :]
        Q = x[:, self.config['MAX_TITLE_LEN']:, :]

        # shape = [Batch_size, # context word, # query word, EMD_DIM]
        shape = (x.size(0), C.size(1), Q.size(1), Q.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)

        S = torch.cat([Ct, Qt, torch.mul(Ct, Qt)], dim=3)
        S = torch.matmul(S, self.W)

        S1 = F.softmax(S, dim=2)
        S2 = F.softmax(S, dim=1)

        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        # [batch_size, title_len, 4*emb_dim]
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)

        X = out.transpose(1, 2)
        X = self.resizer(X)

        for enc in self.model_enc_blks: X = enc(X)

        X = X.view((X.size(0), -1))
        X = self.fc_final(X)
        X = torch.sigmoid(X)

        return X


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception(
                "Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        Wo = torch.empty(D, Dv * Nh)
        Wqs = [torch.empty(D, Dk) for _ in range(Nh)]
        Wks = [torch.empty(D, Dk) for _ in range(Nh)]
        Wvs = [torch.empty(D, Dv) for _ in range(Nh)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(Nh):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x):
        WQs, WKs, WVs = [], [], []
        sqrt_dk_inv = 1 / math.sqrt(Dk)
        x = x.transpose(1, 2)
        for i in range(Nh):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(Nh):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_dk_inv)
            # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
            out = F.softmax(out, dim=2)
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)

class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        D = HID_DIM
        freqs = torch.Tensor(
            [10000 ** (-i / D) if i % 2 == 0 else -10000 ** ((1 - i) / D) for i in range(D)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(D)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(D, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int, hid_dim: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(
            ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length)
        self.normb = nn.LayerNorm([D, length])
        self.norms = nn.ModuleList(
            [nn.LayerNorm([D, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([D, length])
        self.L = conv_num

    def forward(self, x):
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        out = self.self_att(out)
        out = out + res
        out = F.dropout(out, p=dropout)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=dropout)
        return out
