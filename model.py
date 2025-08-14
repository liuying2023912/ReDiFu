import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def gelu(x, dataset='IEMOCAP'):
    if dataset == 'IEMOCAP':
        coefficient = 0.5
    else:
        coefficient = 0.66
    return coefficient * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512, base=10000, dataset='IEMOCAP'):
        super().__init__()
        self.dataset = dataset
        self.dim = dim

        if dataset == 'MELD':
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        else:
            self.max_len = max_len
            self.base = base

            if dim % 2 != 0:
                self.rotary_dim = dim - 1
            else:
                self.rotary_dim = dim

            inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
            self.register_buffer('inv_freq', inv_freq)

            self.modality_scales = nn.Parameter(torch.ones(3))

    def _apply_rotary_pos_emb(self, x, cos_pos, sin_pos):
        x_rotary = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        x1 = x_rotary[..., 0::2]
        x2 = x_rotary[..., 1::2]

        rotated_x1 = x1 * cos_pos - x2 * sin_pos
        rotated_x2 = x1 * sin_pos + x2 * cos_pos

        rotated_x = torch.zeros_like(x_rotary)
        rotated_x[..., 0::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2

        if self.rotary_dim < self.dim:
            rotated_x = torch.cat([rotated_x, x_pass], dim=-1)

        return rotated_x

    def forward(self, x, speaker_emb=None, modality_ids=None):
        if self.dataset == 'MELD':
            L = x.size(1)
            pos_emb = self.pe[:, :L]
            x = x + pos_emb
            if speaker_emb is not None:
                x = x + speaker_emb
            return x
        else:
            batch_size, seq_len, dim = x.size()

            positions = torch.arange(seq_len, device=x.device).float()

            if modality_ids is not None:
                scales = self.modality_scales[modality_ids]
                positions = positions.unsqueeze(0) * scales
            else:
                positions = positions.unsqueeze(0).expand(batch_size, -1)

            freqs = torch.outer(positions.flatten(), self.inv_freq)
            freqs = freqs.view(batch_size, seq_len, -1)

            cos_pos = torch.cos(freqs)
            sin_pos = torch.sin(freqs)

            x = self._apply_rotary_pos_emb(x, cos_pos, sin_pos)

            if speaker_emb is not None:
                if speaker_emb.size(1) == 1:
                    speaker_emb = speaker_emb.expand(-1, seq_len, -1)
                x = x + speaker_emb

            return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2, dataset='IEMOCAP'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dataset = dataset
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(gelu(self.w_1(self.layer_norm(x)), self.dataset))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.2):
        super().__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, head_count, -1, -1)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                   head_count * dim_per_head)
        output = self.linear(context)
        return output, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, dataset='IEMOCAP'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, dataset)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if iter != 0:
            inputs_b_norm = self.layer_norm(inputs_b)
        else:
            inputs_b_norm = inputs_b

        if inputs_a.equal(inputs_b):
            context, atten_score = self.self_attn(inputs_b_norm, inputs_b_norm, inputs_b_norm, mask=mask)
        else:
            inputs_a_norm = self.layer_norm(inputs_a)
            context, atten_score = self.self_attn(inputs_a_norm, inputs_a_norm, inputs_b_norm, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out), atten_score


class LearnableDelta(nn.Module):
    def __init__(self, d_model, window_size=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size // 2, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_t = x.transpose(1, 2)
        delta = self.conv(x_t).transpose(1, 2)
        delta = self.norm(delta)
        gate = self.gate(x)
        return x + self.alpha * gate * delta


class ContextAwareGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 2)
        self.w = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x, delta):
        gate_input = torch.cat([x, delta], dim=-1)
        gate_input = self.norm(gate_input)
        gate = self.w(gate_input)
        return gate * delta + (1 - gate) * x


class DifferentialAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        self.lambda_ = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mask=None):
        B, L, D = x.size()
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_prev = torch.roll(k, shifts=1, dims=2)
        attn_current = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_prev = torch.matmul(q, k_prev.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_current = attn_current.masked_fill(mask == 0, -1e9)
            attn_prev = attn_prev.masked_fill(mask == 0, -1e9)

        attn1 = F.softmax(attn_current, dim=-1)
        attn2 = F.softmax(attn_prev, dim=-1)

        diff_attn = attn1 - attn2
        diff_attn = self.lambda_ * self.dropout(diff_attn)

        out = torch.matmul(diff_attn, v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class DifferentialTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.2, dataset='IEMOCAP'):
        super().__init__()
        self.self_attn = DifferentialAttention(d_model, heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, dataset)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.last_attention = None

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, mask))
        x = self.norm2(x + self.feed_forward(x))
        return x


class DifferentialTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.2, window_size=3, dataset='IEMOCAP'):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.dataset = dataset
        self.pos_emb = PositionalEncoding(d_model, dataset=dataset)
        self.dropout = nn.Dropout(dropout)
        self.delta_net = LearnableDelta(d_model, window_size)
        self.context_gate = ContextAwareGate(d_model)
        self.internal_transformer = nn.ModuleList([
            DifferentialTransformerEncoderLayer(d_model, heads, d_ff, dropout, dataset)
            for _ in range(layers)
        ])
        self.pre_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.last_delta_x = None
        self.last_attention_score = None

    def forward(self, x, mask, speaker_emb=None):
        x_with_pos = self.pos_emb(x, speaker_emb)
        x_with_pos = self.dropout(x_with_pos)

        delta_x = self.delta_net(x)
        processed_delta = self.context_gate(x_with_pos, delta_x)
        processed_delta = self.pre_norm(processed_delta + x_with_pos)

        for idx, layer in enumerate(self.internal_transformer):
            processed_delta = layer(processed_delta, mask)

        if self.dataset == 'IEMOCAP':
            output = x_with_pos + 0.5 * processed_delta
        else:
            output = x_with_pos + processed_delta
        self.last_delta_x = delta_x

        return output, self.last_attention_score


class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation=True, num_relation=-1,
                 relation_dim=10):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.relation = relation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if self.relation:
            self.relation_embedding = relation_embedding
            self.a = nn.Parameter(torch.empty(size=(2 * out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        if self.relation:
            long_adj = adj.clone().type(torch.LongTensor).to(h.device)
            relation_one_hot = self.relation_embedding(long_adj)
            a_input = torch.cat([a_input, relation_one_hot], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention_score = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        h_prime = self.layer_norm(h_prime)
        if self.concat:
            return F.gelu(h_prime), attention_score
        else:
            return h_prime, attention_score

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(1)
        B = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)


class RGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=2, num_relation=-1):
        super(RGAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,
                                                             relation=True, num_relation=num_relation) for _ in
                                         range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False,
                                           relation=True, num_relation=num_relation)
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj):
        residual = x
        x = F.dropout(x, self.dropout, training=self.training)
        attened_outputs = []
        attention_weights = []
        for att_module in self.attentions:
            att_out, att_w = att_module(x, adj)
            attened_outputs.append(att_out)
            attention_weights.append(att_w)
        x = torch.cat(attened_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x, att_w = self.out_att(x, adj)
        attention_weights.append(att_w)
        x = F.gelu(x)
        x = self.fc(x)
        x = x + residual
        x = self.layer_norm(x)
        return x, attention_weights


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., num_layers=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features, hidden_features))
        self.layers.append(act_layer())
        self.layers.append(nn.Dropout(drop))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(act_layer())
            self.layers.append(nn.Dropout(drop))

        self.layers.append(nn.Linear(hidden_features, out_features))
        self.layers.append(nn.Dropout(drop))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_other):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qkv_other = self.qkv(x_other).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_other, v_other = qkv_other[1], qkv_other[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        attn_other = (q @ k_other.transpose(-2, -1)) * self.scale
        attn_other = attn_other.softmax(dim=-1)
        attn_other = self.attn_drop(attn_other)
        x_other_out = (attn_other @ v_other).transpose(1, 2).reshape(B, N, C)
        x_other_out = self.proj(x_other_out)
        x_other_out = self.proj_drop(x_other_out)

        return x_out, x_other_out


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        gate = self.gate(torch.cat([x, y], dim=-1))
        return x * gate + y * (1 - gate)


class MutualFormer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MutualFormer, self).__init__()
        self.norm_t = norm_layer(dim)
        self.norm_a = norm_layer(dim)
        self.norm_v = norm_layer(dim)

        self.attn_t = MultiAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.attn_a = MultiAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.attn_v = MultiAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, proj_drop=drop)

        self.gate_t = GatedFusion(dim)
        self.gate_a = GatedFusion(dim)
        self.gate_v = GatedFusion(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_t = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_a = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_v = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, t, a, v, mask):
        t_norm = self.norm_t(t)
        a_norm = self.norm_a(a)
        v_norm = self.norm_v(v)

        _, t_cross_a_out = self.attn_t(t_norm, a_norm)
        _, t_cross_v_out = self.attn_t(t_norm, v_norm)
        t_out = t + self.drop_path(self.mlp_t(self.gate_t(t, t_cross_a_out + t_cross_v_out)))

        _, a_cross_t_out = self.attn_a(a_norm, t_norm)
        _, a_cross_v_out = self.attn_a(a_norm, v_norm)
        a_out = a + self.drop_path(self.mlp_a(self.gate_a(a, a_cross_t_out + a_cross_v_out)))

        _, v_cross_t_out = self.attn_v(v_norm, t_norm)
        _, v_cross_a_out = self.attn_v(v_norm, a_norm)
        v_out = v + self.drop_path(self.mlp_v(self.gate_v(v, v_cross_t_out + v_cross_a_out)))

        return t_out, a_out, v_out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1, dataset='IEMOCAP'):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model, dataset=dataset)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, dataset) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b, mask, speaker_emb=None):
        if speaker_emb is not None:
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
        for i in range(self.layers):
            x_b, atten_score = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        return x_b, atten_score


class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Transformer_Based_Model, self).__init__()
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dataset = dataset

        padding_idx = 2 if self.n_speakers == 2 else 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx=padding_idx)

        global relation_embedding
        relation_embedding = nn.Embedding(6, 10)

        self.textf_input = nn.Linear(D_text, hidden_dim)
        self.acouf_input = nn.Linear(D_audio, hidden_dim)
        self.visuf_input = nn.Linear(D_visual, hidden_dim)

        self.a_a = DifferentialTransformerEncoder(
            d_model=hidden_dim,
            d_ff=hidden_dim,
            heads=n_head,
            layers=1,
            dropout=dropout,
            dataset=dataset
        )
        self.v_v = DifferentialTransformerEncoder(
            d_model=hidden_dim,
            d_ff=hidden_dim,
            heads=n_head,
            layers=1,
            dropout=dropout,
            dataset=dataset
        )

        self.agate = EnhancedFilterModule(hidden_dim)
        self.vgate = EnhancedFilterModule(hidden_dim)

        self.gatTer = RGAT(hidden_dim, hidden_dim, num_relation=4)
        self.gatT = RGAT(hidden_dim, hidden_dim, num_relation=4)

        if self.dataset == 'IEMOCAP':
            self.mutual_former = MutualFormer(
                dim=hidden_dim,
                num_heads=n_head // 2,
                drop=dropout,
                attn_drop=dropout
            )
        else:  # MELD
            # MELD使用n_head的头数
            self.mutual_former = MutualFormer(
                dim=hidden_dim,
                num_heads=n_head,
                drop=dropout,
                attn_drop=dropout
            )
        if self.dataset == 'IEMOCAP':
            self.t_output_layer = nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
            self.a_output_layer = nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
            self.v_output_layer = nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        else:
            self.t_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
            self.a_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
            self.v_output_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, Self_semantic_adj, Cross_semantic_adj, Semantic_adj):
        device = qmask.device

        spk_idx = torch.argmax(qmask, -1)
        padding_value = 2 if self.n_speakers == 2 else 9
        max_len = spk_idx.size(1)
        indices = torch.arange(max_len, device=device).expand(spk_idx.size(0), -1)
        padding_mask = indices >= dia_len.unsqueeze(1)
        spk_idx = spk_idx.clone()
        spk_idx[padding_mask] = padding_value
        spk_embeddings = self.speaker_embeddings(spk_idx)

        text_feat = self.textf_input(textf.permute(1, 0, 2))
        text_feat, _ = self.gatTer(text_feat, Cross_semantic_adj)
        text_feat, _ = self.gatT(text_feat, Self_semantic_adj)

        sub_log_prog = []
        if visuf is not None and acouf is not None:
            acouf_feat = self.acouf_input(acouf.permute(1, 0, 2))
            visuf_feat = self.visuf_input(visuf.permute(1, 0, 2))

            acouf_feat, _ = self.a_a(acouf_feat, mask=u_mask, speaker_emb=spk_embeddings)
            visuf_feat, _ = self.v_v(visuf_feat, mask=u_mask, speaker_emb=spk_embeddings)

            acouf_feat = self.agate(acouf_feat)
            visuf_feat = self.vgate(visuf_feat)

            text_fused, audio_fused, visual_fused = self.mutual_former(text_feat, acouf_feat, visuf_feat, u_mask)
            if self.dataset == 'IEMOCAP':
                t_logits = self.t_output_layer(text_fused + text_feat)
                a_logits = self.a_output_layer(audio_fused + acouf_feat)
                v_logits = self.v_output_layer(visual_fused + visuf_feat)
            else:
                t_logits = self.t_output_layer(text_fused)
                a_logits = self.a_output_layer(audio_fused)
                v_logits = self.v_output_layer(visual_fused)

            if self.dataset == 'IEMOCAP':
                all_final_out = 1.0 * t_logits + 1.0 * a_logits + 0.4 * v_logits
            else:
                all_final_out = 3.0 * t_logits + 1.0 * a_logits + 0.3 * v_logits

            sub_log_prog.append(F.log_softmax(t_logits, dim=-1))
            sub_log_prog.append(F.log_softmax(a_logits, dim=-1))
            sub_log_prog.append(F.log_softmax(v_logits, dim=-1))

        else:
            t_logits = self.t_output_layer(text_feat)
            all_final_out = t_logits
            sub_log_prog.append(F.log_softmax(t_logits, dim=-1))

        all_log_prob = F.log_softmax(all_final_out, dim=-1)
        all_prob = F.softmax(all_final_out, dim=-1)

        return sub_log_prog, all_log_prob, all_prob, all_final_out