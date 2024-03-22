# modules.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import optim
from torch.autograd import Variable

class Path(nn.Module):
    """convolution to encode every paths beween an entity pair"""
    def __init__(self, input_dim, num_symbols, use_pretrain=True, embed_path='', dropout=0.5, k_sizes = [3], k_num=100):
        '''
        Parameters:
        input_dim: size of relation/entity embeddings
        num_symbols: total number of entities and relations
        use_pretraIn: use pretrained KB embeddings or not
        '''
        super(Path, self).__init__()
        self.symbol_emb = nn.Embedding(num_symbols + 1, input_dim, padding_idx=num_symbols)
        self.k_sizes = k_sizes
        self.k_num = k_num

        if use_pretrain:
            emb_np = np.loadtxt(embed_path)
            self.symbol_emb.weight.data.copy_(torch.from_numpy(emb_np))
            self.symbol_emb.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1,self.k_num, (k, input_dim)) for k in self.k_sizes])

        self.dropout = nn.Dropout(dropout)

    def forward(self, path):
        '''
        Inputs:
        path: batch * max_len(7)
        '''
        path = self.symbol_emb(path)
        path = path.unsqueeze(1) # (B, 1, W, D)

        convs = [F.relu(conv(path)).squeeze(3) for conv in self.convs] # every element (B, 100, W-(k-1))
        pools = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in convs]

        path = torch.cat(pools, 1) # (B, num_k * c_out)
        path = self.dropout(path)

        return path

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)

        self.proj = nn.Linear(n_head*d_v, d_model)
        init.xavier_normal_(self.proj.weight)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        else:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        # enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class ContextAwareEncoder(nn.Module):
    """Use self-attention here"""
    def __init__(self, num_layers, d_model, d_inner_hid, n_head, d_k, d_v, dropout = 0.1):
        super(ContextAwareEncoder, self).__init__()
        self.num_layers = num_layers
        #
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout) for _ in range(self.num_layers)])

    def forward(self, elements, enc_slf_attn_mask=None):
        enc_output = elements
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)

        return enc_output

class QueryEncoder(nn.Module):
    """docstring for QueryEncoder"""
    def __init__(self, input_dim, process_step=4):
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        # self.batch_size = batch_size
        self.process = nn.LSTMCell(input_dim, 2*input_dim)

        # initialize the hidden states, TODO: try to train the initial state
        # self.h0 = Variable(torch.zeros(self.batch_size, 2*input_dim)).cuda()
        # self.c0 = Variable(torch.zeros(self.batch_size, 2*input_dim)).cuda()

    def forward(self, support, query):
        '''
        support: (few, support_dim)
        query: (batch_size, query_dim)
        support_dim = query_dim

        return:
        (batch_size, query_dim)
        '''
        assert support.size()[1] == query.size()[1]

        if self.process_step == 0:
            return query

        batch_size = query.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2*self.input_dim)).cuda()
        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:,:self.input_dim] # (batch_size, query_dim)
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support) # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        # return h_r_[:, :self.input_dim]
        return h

if __name__ == '__main__':
    # test code for modules
    support_encoder = ContextAwareEncoder(2, 100, 200, 4, 25, 25)
    support = Variable(torch.randn(128, 200,100))
    # support = support.unsqueeze(0)
    print(support_encoder(support).size())


# matcher.py
    
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
    """
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support) # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)
        # support_g = support
        # query_g = query

        query_f = self.query_encoder(support_g, query_g) # 128 * 100

        # cosine similarity
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

        return matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1) # tanh
        support = torch.cat((support_left, support_right), dim=-1) # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(torch.abs(support_expand - query))).squeeze()
        return distances



class RescalMatcher(nn.Module):
    """
    Matching based on KB Embeddings
    """
    def __init__(self, embed_dim, num_ents, num_rels, use_pretrain=True, ent_embed=None, rel_matrices=None, dropout=0.1, attn_layers=1, n_head=4, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(RescalMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.ent_emb = nn.Embedding(num_ents + 1, embed_dim, padding_idx=num_ents)
        self.rel_matrices = nn.Embedding(num_rels + 1, embed_dim * embed_dim, padding_idx=num_rels)

        self.aggregate = aggregate

        self.gcn_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        # self.gcn_self_r = nn.Parameter(torch.FloatTensor(1, self.embed_dim))

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal(self.gcn_w.weight)
        init.constant(self.gcn_b, 0)

        if use_pretrain:
            print('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
            self.ent_emb.weight.data.copy_(torch.from_numpy(ent_embed))
            self.rel_matrices.weight.data.copy_(torch.from_numpy(rel_matrices))
            if not finetune:
                print('FIX KB EMBEDDING')
                self.ent_emb.weight.requires_grad = False
                self.rel_matrices.weight.requires_grad = False

        d_model = embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        # self.slf_neighbor = ContextAwareEncoder(attn_layers, d_model, d_inner_hid, n_head, d_k, d_v, dropout)

        # extra parameters for Siamese Neural Networks
        # self.siamese = nn.Linear(d_model, 1)
        # init.xavier_normal(self.siamese.weight)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout(self.rel_matrices(relations)) # (batch, 200, embed_dim*embed_dim)
        ent_embeds = self.dropout(self.ent_emb(entities)) # (batch, 200, embed_dim)

        batch_size = rel_embeds.size()[0]
        max_neighbors = rel_embeds.size()[1]
        rel_embeds = rel_embeds.view(-1, self.embed_dim, self.embed_dim)
        ent_embeds = ent_embeds.view(-1, self.embed_dim).unsqueeze(2)

        # concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)
        concat_embeds = torch.bmm(rel_embeds, ent_embeds).squeeze().view(batch_size,max_neighbors,-1)
        # print concat_embeds.size()

        # concat_embeds = self.slf_neighbor(concat_embeds)

        out = self.gcn_w(concat_embeds) + self.gcn_b # (batch, 200, embed_dim)
        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        # out = F.relu(out)
        out = F.tanh(out)
        # out = F.sigmoid(out)
        return out

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        if query_meta == None:
            support = self.dropout(self.symbol_emb(support)).view(-1, 2*self.embed_dim)
            query = self.dropout(self.symbol_emb(query)).view(-1, 2*self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            matching_scores = torch.matmul(query_f, support_g.t())
            if self.aggregate == 'max':
                query_scores = torch.max(matching_scores, dim=1)[0]
            elif self.aggregate == 'mean':
                query_scores = torch.mean(matching_scores, dim=1)
            elif self.aggregate == 'sum':
                query_scores = torch.sum(matching_scores, dim=1)
            return query_scores


        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1) # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1) # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support) # 1 * 100
        query_g = self.support_encoder(query)

        query_f = self.query_encoder(support_g, query_g) # 128 * 100

        # cosine similarity
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

        # if self.aggregate == 'max':
        #     matching_scores = torch.max(matching_scores, dim=1)[0]
        # elif self.aggregate == 'mean':
        #     matching_scores = torch.mean(matching_scores, dim=1)
        # elif self.aggregate == 'sum':
        #     matching_scores = torch.sum(matching_scores, dim=1)

        # query_scores = F.sigmoid(query_scores)
        return matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1) # tanh
        support = torch.cat((support_left, support_right), dim=-1) # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(torch.abs(support_expand - query))).squeeze()
        return distances

if __name__ == '__main__':

    query = Variable(torch.ones(64,2).long()) * 100
    support = Variable(torch.ones(40,2).long())
    matcher = EmbedMatcher(40, 200, use_pretrain=False)
    print(matcher(query, support).size())