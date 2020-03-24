import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random

from .attention import *

class Transformer(nn.Module):
    def __init__(self, cnn, vocab_size, config):
        super().__init__()

        self.cnn = cnn
        self.vocab_size = vocab_size
        self.Ic = nn.Linear(self.cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(vocab_size, config['attn_size'])
        self.character_distribution = nn.Linear(config['attn_size'], vocab_size)

        self.transformer = nn.Transformer(
                d_model=config['attn_size'],
                nhead=config['nhead'],
                num_encoder_layers=config['encoder_nlayers'],
                num_decoder_layers=config['decoder_nlayers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
        )
        # if config.get('use_encoder', True):
        #     encoder_layer = TransformerEncoderLayer(self.cnn.n_features, nhead=config['encoder_nhead'], attn_type=config['encoder_attn'], use_FFNN=config['use_FFNN'])
        #     self.encoder = TransformerEncoder(encoder_layer, num_layers=config['encoder_nlayers'])
        # else:
        #     self.encoder = None
        # decoder_layer = TransformerDecoderLayer(self.cnn.n_features, vocab_size, config['attn_size'],
        #                                         nhead_vocab=config['decoder_nhead'], nhead_attn=config['encoder_decoder_nhead'],
        #                                         decoder_attn=config['decoder_attn'], encoder_decoder_attn=config['encoder_decoder_attn'],
        #                                         direct_additive=config['direct_additive'], use_FFNN=config['use_FFNN'])
        # self.decoder = TransformerDecoder(config['attn_size'], decoder_layer, num_layers=config['decoder_nlayers'])


    # def generate_subsquence_mask(self, batch_size, size):
    #     mask = torch.tril(torch.ones(batch_size, size, size)).bool()
    #     return mask

    def forward(self, images, targets, teacher_forcing_ratio=0.5):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param targets: Tensor of [B,L], which should start with <start> and end with <end>
        Return:
            - outputs: [B,L,V]
        '''
        batch_size = images.size(0)

        # Step 1: CNN Feature Extraction
        image_features = self.cnn(images) # [B, C', H', W']
        image_features = image_features.transpose(-2, -1) # [B, C', W', H']
        image_features = image_features.reshape(batch_size, self.cnn.n_features, -1) # [B, C', S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, C']
        image_features = self.Ic(image_features) # [B,S,A]
        image_features = image_features.transpose(0, 1) # [S, B, A]

        targets = F.one_hot(targets, self.vocab_size).to(images.device)

        # Step 2: Encoder forwarding
        # if self.encoder is not None:
        #     image_features, _ = self.encoder(image_features, output_weights=False)

        # Step 3: Decoder forwarding
        # targets = targets.float()
        # max_length = targets.shape[1]
        # attn_mask = self.generate_subsquence_mask(batch_size, max_length).to(targets.device)
        # output, _ = self.decoder(image_features, targets, attn_mask)
        # output = self.character_distribution(output)

        targets = self.Vc(targets.float()).transpose(0, 1) # [L,B,A]
        max_length = targets.size(0)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, max_length).to(targets.device)
        output = self.transformer(image_features, targets, tgt_mask=attn_mask)
        output = self.character_distribution(output.transpose(0,1))
        return output

    def greedy(self, images, start_input, output_weights=False, max_length=10):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param start_input: Tensor of [B,1], which is <start> character in onehot
        Return:
            - outputs: [B,L,V]
            - weights: None #TODO: not implement yet
        '''
        batch_size = images.size(0)

        # Step 1: CNN Feature Extraction
        image_features = self.cnn(images) # [B, C', H', W']
        image_features = image_features.transpose(-2, -1) # [B, C', W', H']
        image_features = image_features.reshape(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.transpose(1,2) # [B,S,C']
        image_features = self.Ic(image_features).transpose(0, 1) # [S,B,A]

        predicts = F.one_hot(start_input, self.vocab_size).to(start_input.device)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, max_length).to(predicts.device)
        for t in range(max_length):
            targets = self.Vc(predicts.float()).transpose(0, 1) # [T,B,A]
            output = self.transformer(image_features, targets, tgt_mask=attn_mask[:t+1, :t+1])
            output = output.transpose(0, 1) # [B,T,A]
            output = self.character_distribution(output[:,[-1]]) # [B,1,V]
            output = F.softmax(output, -1)
            index = output.topk(1, -1)[1]
            output = torch.zeros_like(output).long()
            output.scatter_(-1, index, 1)
            predicts = torch.cat([predicts, output], dim=1)
        return predicts[:,1:], None #TODO: return weight

    # def greedy(self, images, start_input, output_weights=False, max_length=10):
    #     '''
    #     Inputs:
    #     :param images: [B,C,H,W]
    #     :param start_input: Tensor of [B,1,V], which is <start> character in onehot
    #     Return:
    #         - outputs: [B,L,V]
    #         - weights: None #TODO: not implement yet
    #     '''
    #     batch_size = images.size(0)

    #     # Step 1: CNN Feature Extraction
    #     image_features = self.cnn(images) # [B, C', H', W']
    #     image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
    #     image_features = image_features.transpose(1,2) # [B,S,C']

    #     # Step 2: Encoder forwarding
    #     if self.encoder is not None:
    #         image_features, weight_encoder = self.encoder(image_features, output_weights=output_weights)
    #     else:
    #         weight_encoder = None
    #         # image_features: [S,B,C']
    #         # weight_encoder: None or list of **num_layers** tensors of shape [B,S,S]
    #     # Step 3: Decoder forwarding
    #     predicts = start_input.float()
    #     for t in range(max_length):
    #         attn_mask = self.generate_subsquence_mask(batch_size, predicts.size(1))
    #         output, weight_decoder = self.decoder(image_features, predicts, attn_mask, output_weights=output_weights)
    #         output = self.character_distribution(output[:,[-1]])
    #         output = F.softmax(output, -1)
    #         index = output.topk(1, -1)[1]
    #         output = torch.zeros_like(output)
    #         output.scatter_(-1, index, 1)
    #         predicts = torch.cat([predicts, output], dim=1)


    #     if output_weights:
    #         # list of 2-tuple to tuple of two list
    #         # text_weight, en_de_weight = zip(*weight_decoder)
    #         # en_de_weight = torch.cat(en_de_weight, dim=1)
    #         # text_weight = torch.cat(text_weight, dim=1)

    #         # weight_decoder: None or list of **num_layers** tuples, each tuple is ([B,T,T], [B,T,S])
    #         return predicts[:,1:], (weight_encoder, weight_decoder)
    #     else:
    #         return predicts[:,1:], None

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, batch_first=False, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1) # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0)
        if batch_first:
            pe.transpose_(0, 1)
        self.register_buffer('pe', pe)
        self.pe.ndimension()

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_width=100):
        super(PositionalEncoding2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_width, d_model)
        position = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, attn_size, decoder_layer, num_layers=1, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, image_features, tgt, attn_mask=None, output_weights=False):

        output = tgt
        weights = []
        for i in range(self.num_layers):
            output, weight = self.layers[i](image_features, tgt, attn_mask, output_weights=output_weights)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

        if output_weights:
            return output, weights
        else:
            return output, None

class TransformerDecoderLayer(nn.Module):
    def __init__(self, cnn_features, vocab_size, attn_size,
                 nhead_vocab, nhead_attn, decoder_attn, encoder_decoder_attn,
                 direct_additive=False, use_FFNN=True, dim_feedforward=2048, dropout=0.1):
        '''
        TransformerDecoderLayer
        - cnn_features (int): dimension of features that CNN extracts
        - vocab_size (int): embedding size of character
        - attn_size (int): for AdditiveAttention
        - nhead_vocab (int): number of head for self attention (= 1 if not use multihead)
        - nhead_attn (int): number of head for attention between encoder and decoder (= 1 if not use multihead)
        - decoder_attn (str): type of self attention, should be "additive" or "scale_dot_product"
        - encoder_decoder_attn (str): type of attention between encoder and decoder, should be "additive" or "scale_dot_product"
        - direct_additive (bool): parameter of Additive Attention (see Notes)
        - use_FFNN (bool): Use PositionWiseFeedForward component
        Notes:
        - MultiheadAttention requires the same dimension of queries and keys (i.e. cnn_features == vocab_size), thus
        normally, we could convert dimension by a nn.Linear to attn_size (see self.Wc, self.Uc). However additive attention
        also convert dimension (see AdditiveAttention implementation), this would make model too complicated.
          - direct_additive=True: Ignore nhead_attn and use AdditiveAttention only
          - direct_additive=False: Convert queries and values to attn_size and use MultiheadAttention with nhead_attn heads
        '''
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = get_attention(decoder_attn, vocab_size, nhead_vocab)

        self.Ic = nn.Linear(cnn_features, attn_size)
        self.Vc = nn.Linear(vocab_size, attn_size)
        self.encoder_decoder_attn = get_attention(encoder_decoder_attn, attn_size, nhead_attn)

        self.norm1 = nn.LayerNorm(vocab_size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(attn_size)
        self.dropout2 = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(attn_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, attn_size)
        self.norm3 = nn.LayerNorm(attn_size)
        self.dropout3 = nn.Dropout(dropout)

        self.attn_size = attn_size

    def forward(self, image_features, tgt, attn_mask=None, output_weights=False):
        tgt2, weight_text = self.self_attn(tgt, tgt, tgt, attn_mask, output_weights)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.Vc(tgt)
        image_features = self.Ic(image_features)

        tgt2, weight_attn = self.encoder_decoder_attn(tgt, image_features, image_features, None, output_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if output_weights:
            return tgt, (weight_text, weight_attn)
        else:
            return tgt, None

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers=1, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, img_features, output_weights=False):
        output = img_features

        weights = []
        for i in range(self.num_layers):
            output, weight = self.layers[i](output, output_weights)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

        if output_weights:
            return output, weights
        else:
            return output, None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, attn_size, nhead, attn_type, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_size = attn_size
        self.self_attn = get_attention(attn_type, attn_size, nhead)
        self.norm1 = nn.LayerNorm(attn_size)
        self.dropout1 = nn.Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(attn_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, attn_size)
        self.norm2 = nn.LayerNorm(attn_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, img_features, output_weights=False):
        img_features2, weight = self.self_attn(img_features, img_features, img_features, output_weights=output_weights)
        img_features = img_features + self.dropout1(img_features2)
        img_features = self.norm1(img_features)

        img_features2 = self.linear2(self.dropout(F.relu(self.linear1(img_features))))
        img_features = img_features + self.dropout2(img_features2)
        img_features = self.norm2(img_features)

        if output_weights:
            return img_features, weight
        else:
            return img_features, None

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# if __name__ == "__main__":
#     attn_size = 256
#     vocab_size = 100
#     feature_size = 200
#     batch_size = 2
#     model = TransformerModel(feature_size, vocab_size, attn_size)
#     print(model)
#     with torch.no_grad():
#         tgt = torch.ones(5, batch_size, vocab_size)
#         image_features = torch.rand(150, batch_size, feature_size)
#         outputs = model.forward(image_features, tgt, None, None, None)
#         outputs = F.softmax(outputs, -1)
#     print('outputs:', outputs.size())
#     print(outputs)
