import sys
import os
from datetime import datetime
import math, copy, time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import _pickle as cPickle
import collections
import evaluate as eva
from sparsemax import Sparsemax
import torch.autograd as autograd
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from allennlp.modules.elmo import Elmo, batch_to_ids
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='/Desktop/DeepWMaxSAT/')
parser.add_argument('--dataset', type=str, default='res14')
parser.add_argument('--setting', type=str, default='dms')
parser.add_argument('--ispre', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=1024, help='Input dimension.')
parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden dimension.')
parser.add_argument('--output_dim', type=int, default=5, help='output classes.')
parser.add_argument('--outputrel_dim', type=int, default=2, help='output relation classes.')
parser.add_argument('--attention_dim', type=int, default=12, help='number of attention heads.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--sample', type=float, default=0.5, help='Sample rate.')
parser.add_argument('--optimizer', type=str, default='adadelta', help='Optimizer.')
#parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.') # adam
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--batch', type=int, default=20, help='Batch size.')
parser.add_argument('--pre_ent_epoch', type=int, default=10, help='Number of pre-training epochs.')
parser.add_argument('--pre_epoch', type=int, default=50, help='Number of jointly pre-training epochs.')
parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=50, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--sch', type=float, default=1.0, help='Scheduled sampling.')
parser.add_argument('--draw', type=str, default='smp', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
args = parser.parse_args()
opt = vars(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, words, label, mask, parse):
        self.words = words
        self.label = label
        self.mask = mask
        self.parse = parse
        #self.ind_dic = idic
        self.ntokens = torch.tensor(self.mask != 0, dtype=torch.float).data.sum()


def data_batch(words, labels, parse, batchlen):
    assert len(words) == len(labels)
    word_batches = [words[x: x + batchlen] for x in range(0, len(words), batchlen)]
    label_batches = [labels[x: x + batchlen] for x in range(0, len(labels), batchlen)]
    parse_batches = [parse[x : x + batchlen] for x in range(0, len(parse), batchlen)]
    for wrd, label, parse in zip(word_batches, label_batches, parse_batches):
        batch_max_len = max([len(s) for s in wrd])
        #wrd = pad_sequences(wrd, maxlen=batch_max_len, dtype="str", truncating="post", padding="post")
        label = pad_sequences(label, maxlen=batch_max_len, value=0, dtype="long", truncating="post", padding="post")
        mask = [[int(i != '0') for i in ii] for ii in wrd]
        mask = pad_sequences(mask, maxlen=batch_max_len, value=0, dtype="long", truncating="post", padding="post")
        label, mask = torch.LongTensor(label), torch.LongTensor(mask)
        yield Batch(wrd, label, mask, parse)


class BaseModel(nn.Module):
    def __init__(self, cuda):
        super(BaseModel, self).__init__()
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, scalar_mix_parameters=[0.0, 0.0, 1.0])
        if cuda:
            self.cuda()

    def forward(self, batch_words):
        pass
        character_ids = batch_to_ids(batch_words).to(device)
        embeddings = self.elmo(character_ids)['elmo_representations'][0]
        masks = self.elmo(character_ids)['mask'][0]
        return embeddings


class ModelGru(nn.Module):
    def __init__(self, base, d_in, d_h, d_chunk, sample):
        super(ModelGru, self).__init__()
        self.base = base
        self.d_in = d_in
        self.d_h = d_h
        self.gru = nn.GRU(d_in, d_h, batch_first=True, dropout=0.1, bidirectional=True)
        self.Wx = nn.Linear(2 * d_h, d_chunk)
        self.sample = sample

    def forward(self, batch_words, parse, train):
        emb = self.base(batch_words)
        rnn_hidden, _ = self.gru(emb)
        SM = nn.Softmax(dim=2)
        a = self.Wx(rnn_hidden)
        output_chunk = SM(self.Wx(rnn_hidden))
        return output_chunk


class MixingFunc(Function):
    # Apply the Mixing method to the input probabilities.

    @staticmethod
    def forward(ctx, in_S, in_z, in_is_input, in_rule_idx, max_iter, eps, prox_lam):
        B = in_z.size(0)
        n, m = in_S.size(0), in_S.size(1)
        ctx.max_iter, ctx.eps, ctx.prox_lam = max_iter, eps, prox_lam

        ctx.z = torch.zeros(B, n, device=device)
        ctx.S = torch.zeros(n, m, device=device)
        ctx.is_input = torch.zeros(B, n, dtype=torch.float, device=device)
        ctx.rule_idx = torch.zeros(B, m, m, dtype=torch.float, device=device)

        ctx.z[:] = in_z.data
        ctx.S[:] = in_S.data
        ctx.is_input[:] = in_is_input.data
        ctx.rule_idx[:] = in_rule_idx.data

        ctx.g = torch.zeros(B, n, k, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)
        ctx.v_t = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
        ctx.V = torch.zeros(B, n, k, device=device)

        count = 0
        for z_b, is_input_b, rule_idx_b, g_b in zip(ctx.z, ctx.is_input, ctx.rule_idx, ctx.g):
            z_b = z_b.unsqueeze(1).to(device)
            new_z = z_b.clone().to(device)
            #v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
            ctx.V[count] = (- torch.cos(pi * new_z)) * ctx.v_t + torch.sin(pi * new_z) * \
                           (F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))\
                            .mm(torch.diag(torch.ones(k, device=device)) - ctx.v_t.t().mm(ctx.v_t))).to(device)
            ctx.V[count] = torch.mul(ctx.V[count].float(), is_input_b.unsqueeze(1).float()).to(device).float()
            # init vo with random vector
            for i in (is_input_b.squeeze() == 0).nonzero():
                #v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
                i = i.squeeze()
                ctx.V[count] = ctx.V[count].index_add_(0, i, F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1)))

            masked_S_t = rule_idx_b.mm(ctx.S.t()).to(device)
            new_S = masked_S_t[masked_S_t.sum(dim=1) != 0].t().to(device)
            new_g_b = g_b.clone().to(device)
            if new_S.size(1) != 0:
                omega = new_S.t().mm(ctx.V[count]).to(device)

                # while not converged: forward pass coord descent
                for it in range(max_iter):
                    early_stop = 0
                    for i in (is_input_b.squeeze() == 0).nonzero():
                        i = i.squeeze()
                        new_g_b[i] = new_S[i].unsqueeze(0).mm(omega).squeeze() - new_S[i].norm().pow(2) * ctx.V[count][i]
                        v_pre = ctx.V[count][i].clone().to(device)
                        if new_g_b[i].norm()!=0:
                            ctx.V[count][i] = - new_g_b[i] / new_g_b[i].norm()
                        omega += new_S[i].unsqueeze(0).t().mm((ctx.V[count][i] - v_pre).unsqueeze(0))
                # output V: vo --> zo
                for i in (is_input_b.squeeze() == 0).nonzero():
                    i = i.squeeze()
                    cos = - ctx.v_t.mm(ctx.V[count][i].unsqueeze(0).t()).squeeze().to(device)
                    cos = torch.clamp(cos, -1 + ctx.eps, 1 - ctx.eps)
                    new_z[i] = torch.acos(cos) / pi
            ctx.V[count] = F.normalize(ctx.V[count])
            ctx.z[count] = new_z.squeeze()
            ctx.g[count] = new_g_b.clone()
            count += 1
        return ctx.z.clone()

    @staticmethod
    def backward(ctx, dz):
        if torch.any(dz != dz):
            res = torch.zeros_like(dz)
        else:
            #dz[dz!=dz]=0
            dz.clamp(0,1)
            B = dz.size(0)
            ctx.U = torch.zeros(B, n, k, device=device)
            ctx.dz = torch.zeros(B, n, device=device)
            ctx.dz[:] = dz.data
            #print("in dz:", ctx.dz)
            ctx.dV = torch.zeros(B, n, k, device=device)

            dz_new = ctx.dz.clone().to(device)
            count = 0
            for dz_b, is_input_b, dV_b, U_b, z_b, rule_idx_b, g_b in zip(dz_new, ctx.is_input, ctx.dV, ctx.U, ctx.z, ctx.rule_idx, ctx.g):
                # dzo -> dvo
                #print("dz_b_0", dz_b)
                for i in (is_input_b.squeeze() == 0).nonzero():
                    i = i.squeeze()
                    dV_b[i] = (dz_b[i] / pi / (torch.sin(pi * z_b[i]))) * ctx.v_t

                masked_S_t = rule_idx_b.mm(ctx.S.t()).to(device)
                new_S = masked_S_t[masked_S_t.sum(dim=1) != 0].t().to(device)
                #print("new_S:", new_S)
                if new_S.size(1) != 0:
                    Phi_b = (new_S.t()).mm(U_b).to(device)

                    # while not converged: backward pass coord descent
                    for it in range(ctx.max_iter):
                        #early_stop = 0
                        for i in (is_input_b.squeeze() == 0).nonzero():
                            i = i.squeeze()
                            dg = new_S[i].unsqueeze(0).mm(Phi_b).squeeze() - new_S[i].norm().pow(2) * U_b[i] - dV_b[i]
                            P = torch.diag(torch.ones(k, device=device)) - ctx.V[count][i].unsqueeze(0).t().mm(ctx.V[count][i].unsqueeze(0))
                            u_pre = U_b[i].clone().to(device)
                            if g_b[i].norm()!=0:
                                U_b[i] = - (dg / g_b[i].norm()).unsqueeze(0).mm(P).to(device)
                            #else:
                                #U_b[i] = u_pre
                            Phi_b += new_S[i].unsqueeze(0).t().mm((U_b[i] - u_pre).unsqueeze(0)).to(device)

                    sum_uws = torch.zeros(new_S.size(1), k, device=device)
                    for i in (is_input_b.squeeze() == 0).nonzero():
                        i = i.squeeze()
                        sum_ums_step = new_S[i].unsqueeze(0).t().mm(U_b[i].unsqueeze(0))
                        sum_uws += sum_ums_step

                    v_rand = F.normalize(torch.randn(k, 1, device=device, dtype=torch.float).normal_().view(1, -1))
                    dv_dz = pi * (torch.sin(pi * z_b).unsqueeze(1) * ctx.v_t + torch.cos(pi * z_b).unsqueeze(1) *
                                  (v_rand.mm(torch.diag(torch.ones(k, device=device)) - ctx.v_t.t().mm(ctx.v_t))))
                    dz_input = torch.diagonal(new_S.mm(sum_uws).mm(dv_dz.t()))

                    dz_new[count] = torch.mul(dz_input.squeeze(), is_input_b) + torch.mul(dz_b, (1 - is_input_b))
                    #ctx.dz[count][0] = 0
                    ctx.dz[count][11:] = 0
                count += 1
            res = ctx.dz.clone()
            #print("out dz:", ctx.dz.clone())

        return None, res, None, None, None, None, None


class RuleAttention(nn.Module):
    """ Applies attention mechanism on the `context` -- S using the `query` -- z.

    Args:
        dimensions: #var

        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:
         attention = Attention(#var)
         query = torch.randn(#batch, 1, #var)
         context = torch.randn(#batch, #rule, #var)
         attn_scores = attention(query, context)
    """

    def __init__(self, dimensions, attention_type='dot', max_type='sparse'):
        super(RuleAttention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        if max_type not in ['sparse', 'soft']:
            raise ValueError('Invalid max type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False).to(device)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False).to(device)
        self.max_type = max_type
        if self.max_type == 'sparse':
            self.softmax = Sparsemax(dim=-1)
        if self.max_type == 'soft':
            self.softmax = nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh().cuda()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [#batch, 1, #var]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [#batch, #rule, #var]): Data
                over which to apply the attention mechanism.

        Returns:
            * **scores** (:class:`torch.FloatTensor`  [#batch, 1, #rules]):
              Tensor containing attention scores.
        """
        batch_size, output_len, dimensions = query.size() # [#batch, 1, #var]
        query_len = context.size(1) # [#rule]

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions).to(device)
            query = self.linear_in(query).to(device)
            query = query.reshape(batch_size, output_len, dimensions).to(device)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous()).to(device)

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len).to(device)
        attention_weights = self.softmax(attention_scores).to(device)
        attention_weights = attention_weights.view(batch_size, output_len, query_len).to(device)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context).to(device)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2).to(device)
        combined = combined.view(batch_size * output_len, 2 * dimensions).to(device)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined.to(device)).view(batch_size, output_len, dimensions).to(device)
        output = self.tanh(output).to(device)

        return attention_weights.view(batch_size * output_len, query_len)


class MaxsatLayer(nn.Module):
    """Apply a MaxSAT layer to complete the input probabilities.

    Args:
        aux: Number of auxiliary variables.
        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True

    Inputs: (z, is_input)
        **z** of shape `(batch, n)`:
            Float tensor containing the probabilities (must be in [0,1]).
        **is_input** of shape `(batch, n)`:
            Int tensor indicating which **z** is a input.

    Outputs: shape same as "output_chunk"

    """

    def __init__(self, model, in_S, aux=0, max_iter=80, eps=1e-4, prox_lam=1e-2, weight_normalize=True):
        super(MaxsatLayer, self).__init__()
        self.model = model
        self.S = in_S
        self.n, self.m, self.k = in_S.size(0), in_S.size(1), 32
        self.thre = nn.Parameter(torch.tensor(0.6, device=device, requires_grad=True))
        self.r = nn.Parameter(torch.tensor(1., device=device, requires_grad=True))
        self.aux, self.max_iter, self.eps, self.prox_lam = aux, max_iter, eps, prox_lam

    def forward(self, batch_words, parse, train):
        output_batch = self.model(batch_words, parse, train)
        label = torch.argmax(output_batch, dim=2)
        new_output_batch = output_batch.clone()
        z = []
        is_input = []
        for out, tag, parse_sen in zip(output_batch, label, parse):
            for [x, x_pos, y, y_pos, r] in parse_sen:
                if (tag[y] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['nn'] and (x - y == 1)) or \
                    (tag[y] in [1, 2] and x_pos == 'ADJ' and y_pos in ['PROPN', 'NOUN'] and r in ['nsubj']) or \
                        (tag[y] in [1, 2] and y_pos == 'ADJ' and x_pos in ['PROPN', 'NOUN'] and r in ['amod'] and (x - y == 1)):
                    z_1 = out[x].to(device)
                    z_2 = out[y].to(device)
                    z_3 = torch.tensor([x_pos == 'ADJ', x_pos in ['PROPN', 'NOUN'], y_pos == 'ADJ', y_pos in ['PROPN', 'NOUN']]).float().to(device)
                    z_4 = torch.tensor([tag[x-1] in[1,2], tag[x-1] not in[1,2], tag[x-1] in[3,4], tag[x-1] not in[3,4]]).float().to(device)
                    z_5 = torch.tensor([r == 'nn', r == 'conj', r == 'nsubj', r == 'amod']).float().to(device)
                    z.append(torch.cat((z_1, z_2, z_4, z_3, z_5), 0))
                    is_input.append(torch.cat([torch.zeros(5).to(device), torch.ones(n-6).to(device)]))
                if (tag[x] in [3, 4] and y_pos == 'ADJ' and x_pos == 'ADJ' and r in ['conj']) or \
                    (tag[x] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['conj']):
                    z_1 = out[x].to(device)
                    z_2 = out[y].to(device)
                    z_3 = torch.tensor([x_pos == 'ADJ', x_pos in ['PROPN', 'NOUN'], y_pos == 'ADJ', y_pos in ['PROPN', 'NOUN']]).float().to(device)
                    z_4 = torch.tensor([tag[y-1] in[1,2], tag[y-1] not in[1,2], tag[y-1] in[3,4], tag[y-1] not in[3,4]]).float().to(device)
                    z_5 = torch.tensor([r == 'nn', r == 'conj', r == 'nsubj', r == 'amod']).float().to(device)
                    z.append(torch.cat((z_1, z_2, z_4, z_3, z_5), 0))
                    is_input.append(torch.cat([torch.ones(5).to(device), torch.zeros(5).to(device), torch.ones(n-11).to(device)]))

        if z != []:
            z = torch.stack(z).to(device)
            is_input = torch.stack(is_input).to(device)

            if (z!=z).any():
                print("warning!")
            else:
                B = z.size(0)
                z = torch.cat([torch.ones(B, 1, device=device), z, torch.zeros(B, 0, device=device)], dim=1)
                is_input = torch.cat([torch.ones(B, 1, device=device), is_input, torch.zeros(B, 0, device=device)], dim=1)
                query = z[:, 1:11].unsqueeze(1).to(device)
                context = torch.abs(S_t_body[:, 1:11]).repeat(B, 1, 1).to(device)
                rule_attention_sparse = RuleAttention(query.size(2), 'dot', 'sparse')
                attn_weights = rule_attention_sparse(query, context).squeeze().to(device)

                if B == 1:
                    attn_weights = attn_weights.unsqueeze(0).to(device)

                rule_idx_vec = torch.zeros_like(attn_weights).to(device)

                for i in range(B):
                    for j in range(self.m):
                        if torch.all(torch.eq(torch.abs(z[:, 15:])[i], torch.abs(S_t[:, 15:])[j])):
                            if torch.all(torch.eq(torch.abs(z[:, 11:13])[i], torch.abs(S_t[:, 11:13])[j])) or \
                                    torch.all(torch.eq(torch.abs(z[:, 13:15])[i], torch.abs(S_t[:, 13:15])[j])) or\
                                    torch.all(torch.eq(torch.zeros_like(torch.abs(S_t[:, 11:15])[j]), torch.abs(S_t[:, 11:15])[j])):
                                rule_idx_vec[i, j] = 1

                rule_idx_vec *= attn_weights
                rule_idx_vec = F.normalize(rule_idx_vec)
                rule_idx = torch.empty(B, self.m, self.m)
                for i in range(B):
                    rule_idx[i] = torch.diag(rule_idx_vec[i])

                z_new = MixingFunc.apply(self.S, z, is_input, rule_idx, self.max_iter, self.eps, self.prox_lam).to(device)

                z_msout = z_new[:, 1:11].to(device)

                count_a = 0
                count_z = 0

                for out, tag, parse_sen in zip(output_batch, label, parse):
                    for [x, x_pos, y, y_pos, r] in parse_sen:
                        if (tag[y] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['nn'] and (x - y == 1)) or \
                           (tag[y] in [1, 2] and x_pos == 'ADJ' and y_pos in ['PROPN', 'NOUN'] and r in ['nsubj']) or \
                            (tag[y] in [1, 2] and y_pos == 'ADJ' and x_pos in ['PROPN', 'NOUN'] and r in ['amod'] and (x - y == 1)) or \
                            (tag[x] in [3, 4] and y_pos == 'ADJ' and x_pos == 'ADJ' and r in ['conj']) or \
                            (tag[x] in [1, 2] and y_pos in ['PROPN', 'NOUN'] and x_pos in ['PROPN', 'NOUN'] and r in ['conj']):
                                new_output_batch[count_a][x] = z_msout[count_z][0:5]
                                new_output_batch[count_a][y] = z_msout[count_z][5:10]
                                count_z += 1
                    count_a += 1
        features = gate(output_batch, new_output_batch, self.r).to(device)
        '''
        m_label = torch.argmax(new_output_batch, dim=2)
        o_label = torch.argmax(features, dim=2)
        
        for sentence, out, tag, m_tag, o_tag, parse_sen in zip(src, output_batch, label, m_label, o_label, parse):
            if torch.any(o_tag != m_tag):
                print("sen:", sentence)
                print("dnn:", tag)
                print("maxsat:", m_tag)
                if torch.any(tag != o_tag):
                    print("choose: maxsat")
                else:
                    if torch.any(m_tag != o_tag):
                        print("choose: dnn")
                    else:
                        print("choose: none")
        '''
        return features.clone()


def gate(h,x,r):
    features = (r * h + (1-r)*x).to(device)
    features = features.float()
    return features


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


class CRF(nn.Module):
    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        self.gpu = gpu
        self.tagset_size = tagset_size
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        init_transitions[:,START_TAG] = -10000.0
        init_transitions[STOP_TAG,:] = -10000.0
        #init_transitions[:,0] = -10000.0
        #init_transitions[0,:] = -10000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size).to(device)
            cur_partition = log_sum_exp(cur_values, tag_size).to(device)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size).to(device)
            mask_idx = mask_idx.bool()
            masked_cur_partition = cur_partition.masked_select(mask_idx).to(device)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1).to(device)
            partition.masked_scatter_(mask_idx, masked_cur_partition).to(device)
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size).to(device) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size).to(device)
        cur_partition = log_sum_exp(cur_values, tag_size).to(device)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size).to(device)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)

        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            mask = mask.bool()
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0).to(device)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous().to(device) ## (batch_size, seq_len. tag_size)
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1).to(device)
        last_values = last_partition.expand(batch_size, tag_size, tag_size).to(device) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size).to(device)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long().to(device)
        #pad_zero = pad_zero.to(device)
        back_points.append(pad_zero.to(device))
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size).to(device)
        back_points = back_points.transpose(1,0).contiguous().to(device)
        back_points.scatter_(1, last_position, insert_last).to(device)
        back_points = back_points.transpose(1,0).contiguous().to(device)
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.to(device)
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1)).to(device)
            decode_idx[idx] = pointer.detach().view(batch_size).to(device)
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2)*tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx-1]*tag_size + tags[:, idx]

        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size).to(device)
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        end_ids = torch.gather(tags, 1, length_mask - 1).to(device)

        end_energy = torch.gather(end_transition.to(device), 1, end_ids.to(device)).to(device)

        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1).to(device)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size).to(device)  # seq_len * bat_size
        mask = mask.bool()
        tg_energy = tg_energy.masked_select(mask.transpose(1,0)).to(device)

        gold_score = tg_energy.sum().to(device) + end_energy.sum().to(device)
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long().to(device)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size).to(device)

        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        partition = inivalues[:, START_TAG, :].clone()  # bat_size * to_target_size
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                cur_values = cur_values.view(batch_size, tag_size*nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            if idx == 1:
                cur_bp = cur_bp*nbest
            partition = partition.transpose(2,1).to(device)
            cur_bp = cur_bp.transpose(2,1).to(device)
            partition_history.append(partition)
            mask = mask.bool()
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0).to(device)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history,0).view(seq_len, batch_size, tag_size, nbest).transpose(1,0).contiguous() ## (batch_size, seq_len, nbest, tag_size)
        last_position = length_mask.view(batch_size,1,1,1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size*nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        end_bp = end_bp.transpose(2,1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        pointer = end_bp[:, STOP_TAG, :] ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1,0).contiguous()
        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1,0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data/nbest
        for idx in range(len(back_points)-2, -1, -1):
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size*nbest), 1, pointer.contiguous().view(batch_size,nbest))
            decode_idx[idx] = new_pointer.data/nbest
            pointer = new_pointer + pointer.contiguous().view(batch_size,nbest)*mask[idx].view(batch_size,1).expand(batch_size, nbest).long()

        path_score = None
        decode_idx = decode_idx.transpose(1,0).to(device)
        scores = end_partition[:, :, STOP_TAG]
        max_scores,_ = torch.max(scores, 1).to(device)
        minus_scores = scores - max_scores.view(batch_size,1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1).to(device)
        return path_score, decode_idx


class ModelCrf(nn.Module):
    def __init__(self, model, num_tags):
        super(ModelCrf, self).__init__()
        self.model = model
        self.crf = CRF(num_tags, False)
        # num_tags = 5

    def loss(self, mask, batch_labels, batch_words, parse, train):
        features = self.model(batch_words, parse, train).to(device) #B, L, C
        tags = batch_labels.clone().to(device) #B, L
        features = features.float().to(device)
        outs = torch.cat([features, torch.ones(features.size(0), features.size(1),2).to(device)],dim=2).to(device)
        masks = mask.squeeze().to(device) #B, L
        if masks.dim()==1:
            masks=masks.unsqueeze(0)
        loss = self.crf.neg_log_likelihood_loss(outs, masks, tags).to(device)
        return loss

    def forward(self, mask, batch_words, parse, train):
        features = self.model(batch_words, parse, train).to(device) #B, L, C
        features = features.float().to(device)
        outs = torch.cat([features, torch.ones(features.size(0), features.size(1), 2).to(device)], dim=2).to(device)
        masks = mask.squeeze().unsqueeze(0) .to(device) # B, L
        if masks.dim()==1:
            masks=masks.unsqueeze(0)
        scores, tag_seq = self.crf._viterbi_decode(outs, masks)
        tag_seq = [torch.tensor(item.clone()) for sublist in tag_seq for item in sublist]
        return scores, tag_seq


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


class Trainer(object):
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])
        #if self.model == model_crf:
        #    optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])
        #    self.optimizer = lr_decay(optimizer, epoch, self.opt['decay'],  self.opt['lr'])

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, mask, label, batch_words, parse, train):
        self.model.train()
        self.optimizer.zero_grad()

        if self.model in [model_elmo, model_maxsat]:
            output_chunk = self.model(batch_words, parse, train)
            output_chunk = output_chunk.view(-1, output_chunk.size(-1))
            loss_flat = -torch.log(torch.gather(output_chunk.contiguous(), dim=1, \
                                                index=label.contiguous().view(-1, 1)))
            loss = loss_flat.view(*label.size())
            loss = (loss * mask.float()).sum()
            # loss_chunk = self.criterion(output_chunk, label)

        if self.model == model_crf:
            loss = self.model.loss(mask, label, batch_words, parse, train)

        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            model_maxsat.r.clamp_(0, 1)
        return loss.item()

    def predict(self, mask, batch_words, parse, train):
        self.model.eval()
        with torch.no_grad():
            if self.model in [model_elmo, model_maxsat]:
                output_chunk = self.model(batch_words, parse, train)
                preds = torch.argmax(output_chunk, dim=2).squeeze()
            if self.model == model_crf:
                _, preds = self.model(mask, batch_words, parse, train)
            return preds

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])


def pre_train(opt, data_train, data_test, labels_test, f):
    total_loss = 0.0
    tokens = 0
    start = time.time()

    for i, batch in enumerate(data_train):
        loss = trainer_elmo.update(batch.mask.to(device),  batch.label.to(device), batch.words, \
                                   batch.parse, train=True)
        total_loss += loss
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %(i, loss, tokens / elapsed))
        tokens = 0
        start = time.time()
    print("Epoch loss: %f" %total_loss)

    # evaluation
    pred_all = []
    for i, batch in enumerate(data_test):
        label_pred = trainer_elmo.predict(batch.mask.to(device), batch.words, batch.parse, train=False)
        if label_pred.dim() == 0:
            label_pred = label_pred.unsqueeze(0)
        pred_all.append(label_pred)

    precision, recall, f1, report = utils.eva_entity(pred_all, labels_test, label_map)
    f.write("pretraining evaluations on entities: \n")
    f.write("precision, recall, f1: " + str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\n")
    f.write(report + "\n")


def train_elmo(opt, data_train, data_test, labels_test, f):
    total_loss = 0.0
    start = time.time()
    tokens = 0

    for i, batch in enumerate(data_train):
        loss = trainer_elmo.update(batch.mask.to(device),  batch.label.to(device), batch.words, \
                                   batch.parse, train=True)
        total_loss += loss
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss, tokens / elapsed))
        tokens = 0
        start = time.time()

    print("Epoch loss: %f" % total_loss)

    # evaluation
    pred_all = []

    for i, batch in enumerate(data_test):
        label_pred = trainer_elmo.predict(batch.mask.to(device), batch.words, batch.parse, train=False)

        if label_pred.dim() == 0:
            label_pred = label_pred.unsqueeze(0)
        pred_all.append(label_pred)

    precision, recall, f1, report = utils.eva_entity(pred_all, labels_test, label_map)
    f.write("training evaluations on entities: \n")
    f.write("precision, recall, f1: " + str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\n")
    f.write(report + "\n")


def train_maxsat(opt, data_train, data_test, labels_test, f):
    total_loss = 0.0
    start = time.time()
    tokens = 0

    for i, batch in enumerate(data_train):
        loss = trainer_maxsat.update(batch.mask.to(device),  batch.label.to(device), batch.words, \
                                   batch.parse, train=True)
        total_loss += loss
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss, tokens / elapsed))
        tokens = 0
        start = time.time()

    print("Epoch loss: %f" % total_loss)

    # evaluation
    pred_all = []

    for i, batch in enumerate(data_test):
        label_pred = trainer_maxsat.predict(batch.mask.to(device), batch.words, batch.parse, train=False)

        if label_pred.dim() == 0:
             label_pred = label_pred.unsqueeze(0)
        pred_all.append(label_pred)


    precision, recall, f1, report = utils.eva_entity(pred_all, labels_test, label_map)
    f.write("training evaluations on entities: \n")
    f.write("precision, recall, f1: " + str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\n")
    f.write(report + "\n")


def train_crf(opt, data_train, data_test, labels_test, f):
    total_loss = 0.0
    start = time.time()
    tokens = 0

    for i, batch in enumerate(data_train):
        loss = trainer_crf.update(batch.mask.to(device),  batch.label.to(device), batch.words, \
                                   batch.parse, train=True)
        total_loss += loss
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss, tokens / elapsed))
        tokens = 0
        start = time.time()

    print("Epoch loss: %f" % total_loss)

    # evaluation
    pred_all = []

    for i, batch in enumerate(data_test):
        label_pred = trainer_crf.predict(batch.mask.to(device), batch.words, batch.parse, train=False)
        for i in range(len(label_pred)):
            if label_pred[i].dim() == 0:
                label_pred[i] = label_pred[i].unsqueeze(0)
        pred_all.append(label_pred)

        #for i in range(len(label_pred)):
        #    if label_pred[i].dim() == 0:
        #        label_pred[i] = label_pred[i].unsqueeze(0)
        #    pred_all.append(label_pred)

    precision, recall, f1, report = utils.eva_entity(pred_all, labels_test, label_map)
    f.write("training evaluations on entities: \n")
    f.write("precision, recall, f1: " + str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\n")
    f.write(report + "\n")


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


words_train = cPickle.load(open(opt['datapath'] + "data/Aspect_Opinion/words_" + opt['dataset'] + ".train", "rb"))
words_test = cPickle.load(open(opt['datapath'] + "data/Aspect_Opinion/words_" + opt['dataset'] + ".test", "rb"))
labels_train = cPickle.load(open(opt['datapath'] + "data/Aspect_Opinion/labels_" + opt['dataset'] + ".train", "rb"))
labels_test = cPickle.load(open(opt['datapath'] + "data/Aspect_Opinion/labels_" + opt['dataset'] + ".test", "rb"))
emb = cPickle.load(open(opt['datapath'] + "data/Aspect_Opinion/embedding300_" + opt['dataset'], "rb"))
idxs_train = cPickle.load(open(opt['datapath'] +"data/Aspect_Opinion/idx_" + opt['dataset'] + ".train", "rb"))
idxs_test = cPickle.load(open(opt['datapath'] +"data/Aspect_Opinion/idx_" + opt['dataset'] + ".test", "rb"))
pos_train, pos_test = cPickle.load(open(opt['datapath'] +"data/Aspect_Opinion/pos_" + opt['dataset'] + ".tag", "rb"))
parse_train = cPickle.load(open(opt['datapath'] +"data/Aspect_Opinion/parse_pos_" + opt['dataset'] + ".train", "rb"))
parse_test = cPickle.load(open(opt['datapath'] +"data/Aspect_Opinion/parse_pos_" + opt['dataset'] + ".test", "rb"))

pos_dic = ['pad']
posind_train, posind_test = [], []
for pos_line in pos_train:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_train.append(pos_ind)
for pos_line in pos_test:
    pos_ind = []
    for pos in pos_line:
        if pos not in pos_dic:
            pos_dic.append(pos)
        pos_ind.append(pos_dic.index(pos))
    posind_test.append(pos_ind)


label_map = {0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'}

START_TAG = -2
STOP_TAG = -1

S_t = torch.tensor([
    #rule1
    [-1.,   0.,  0.,  1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    [-1.,   0.,  0.,  1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    #rule2
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  1.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,    -1.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0., -1.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0., -1.,  0.,    -1.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0., -1.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  1.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,    -1.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  1.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0., -1.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,    -1.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0., -1.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  1.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    #rule3
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0., -1.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0., -1.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0., -1.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0., -1.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  1.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0., -1.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0., -1.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,    -1.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0., -1.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],[-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0., -1.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],


    #rule4
    [-1.,   0.,  0.,  0.,  1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  1.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  1.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],


    #rule5
    [-1.,   0.,  0.,  1.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],

    [-1.,   0.,  0.,  1.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,  -1.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.]
    ], device=device)

S_t_body = torch.tensor([
    #rule1
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0., -1., 0., -1.,    -1.,  0.,  0.,  0.],

    #rule2
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0., -1.,  0.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0.,  0., -1.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0.,  0.,  0., -1.,     0.,  0.,  0.,  0.,  0.,    0.,  0., -1.,  0.,    -1.,  0.,-1.,  0.,    0., -1.,  0.,  0.],

    #rule3
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0., -1.,  0.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,    0., -1.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],

    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],[-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  1.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],
    [-1.,   0.,  0., -1.,  0.,  0.,     0.,  0.,  0.,  0.,  0.,   -1.,  0.,  0.,  0.,     0., -1., 0., -1.,    0., -1.,  0.,  0.],


    #rule4
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0., -1.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0., -1.,  0.,     -1., 0., 0., -1.,    0.,  0., -1.,  0.],


    #rule5
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0., -1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],

    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.],
    [-1.,   0.,  0.,  0.,  0.,  0.,     0.,  0., -1.,  0.,  0.,    0.,  0.,  0.,  0.,     0.,-1., -1.,  0.,    0.,  0.,  0., -1.]
    ], device=device)

S = S_t.t()
# n: number of rules, m: number of logical variables
n, m, k = S.size(0), S.size(1), 32
S = S * ((.5 / (n + 1 + m)) ** 0.5)
pi = torch.tensor([math.pi], device=device)


# object initialization
model_base = BaseModel(cuda=opt['cuda'])
model_elmo = ModelGru(model_base, opt['input_dim'], opt['hidden_dim'], opt['output_dim'], opt['sample'])
model_elmo.to(device)


trainer_elmo = Trainer(opt, model_elmo)

f = open(opt['datapath']+"result/Aspect-Opinion/bert/" + opt['setting'] + opt['dataset'] + opt['optimizer'] + str(opt['lr']) + str(opt['decay']) + ".txt", "w")

model_maxsat = MaxsatLayer(model_elmo, S)

if opt['setting'] in ['dms', 'maxsat']:
    model_crf = ModelCrf(model_maxsat, num_tags=5)
if opt['setting'] == 'transcrf':
    model_crf = ModelCrf(model_elmo, num_tags=5)


model_maxsat = MaxsatLayer(model_elmo, S)
if opt['setting'] in ['dms', 'maxsat']:
    model_crf = ModelCrf(model_maxsat, num_tags=5)
if opt['setting'] == 'transcrf':
    model_crf = ModelCrf(model_elmo, num_tags=5)

trainer_maxsat = Trainer(opt, model_maxsat)
trainer_crf = Trainer(opt, model_crf)


for epoch in range(opt['epoch']):
    print("epoch number: %d" % epoch)
    f.write("Epoch number: " + str(epoch) + "\n")
    print("evaluation of model: \n")
    f.write("evaluation of model: \n")
    print("r:", model_maxsat.r, "\n ")
    data_train = data_batch(words_train, labels_train, parse_train, opt['batch'])
    data_test = data_batch(words_test, labels_test, parse_test, 1)
    if opt['setting'] in ['dms', 'transcrf']:
        train_crf(opt, data_train, data_test, labels_test, f)
    if opt['setting'] in ['maxsat']:
        train_maxsat(opt, data_train, data_test, labels_test, f)

