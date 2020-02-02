import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class TransformerModel(nn.Module):
    def __init__(self, cnn_features, vocab_size, attn_size):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(attn_size, nhead=1)
        self.transformer = TransformerDecoder(attn_size, decoder_layer, num_layers=1)
        
        self.linear_img = nn.Linear(cnn_features, attn_size)
        self.linear_text = nn.Linear(vocab_size, attn_size)
        
        self.linear_output = nn.Linear(attn_size, vocab_size)
        self.vocab_size = vocab_size
        
        self.positional_encoding_text = PositionalEncoding(vocab_size)
    
    def forward(self, img_features, targets_onehot, targets, targets_length, PAD_CHAR_int):
        img_features = self.linear_img(img_features)
        # Optional: PositionEncoding for imgs
        # here ...
        
        targets_onehot = self.positional_encoding_text(targets_onehot.float())
        targets_onehot = self.linear_text(targets_onehot)
        
        tgt_mask = nn.modules.Transformer.generate_square_subsequent_mask(None, torch.max(targets_length)).to(targets.device) # ignore SOS_CHAR
        tgt_key_padding_mask = (targets == PAD_CHAR_int).squeeze(-1).transpose(0,1).to(targets.device) # [B,T]
        
        outputs = self.transformer.forward(targets_onehot, img_features,
                                           tgt_mask=None,
                                           tgt_key_padding_mask=None)
        # [T, N, d_model]
        outputs = self.linear_output(outputs) # [T, N, vocab_size]
        return outputs
    
    def inference(self, img_features, start_input, max_length=10):
        img_features = self.linear_img(img_features)
        # Optional: PositionEncoding for imgs
        # here ...
        
        outputs = start_input
        
        for t in range(max_length):
            transformer_input = self.linear_text(outputs)
            transformer_input = self.positional_encoding_text(transformer_input)
            output = self.transformer.forward(transformer_input, img_features,
                                              tgt_mask=None, memory_mask=None,
                                              tgt_key_padding_mask=None,
                                              memory_key_padding_mask=None)
            output = self.linear_output(output[[-1]])
            output = F.softmax(output, -1)
            _, index = output.topk(1, -1)
            predict = torch.zeros(1, 1, self.vocab_size).to(img_features.device)
            predict[:,:,index] = 1
            outputs = torch.cat([outputs, predict], dim=0)

        return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, attn_size, decoder_layer, num_layers=1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(attn_size)

    def forward(self, tgt, image_features, tgt_mask=None,
                tgt_key_padding_mask=None):

        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, image_features, tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, attn_size, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.modules.MultiheadAttention(attn_size, nhead, dropout=dropout)
        self.multihead_attn = nn.modules.MultiheadAttention(attn_size, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(attn_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, attn_size)

        self.norm1 = nn.LayerNorm(attn_size)
        self.norm2 = nn.LayerNorm(attn_size)
        self.norm3 = nn.LayerNorm(attn_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.attn_size = attn_size

    def forward(self, tgt, image_features, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, image_features, image_features, attn_mask=None)[0] # TODO: get weights here
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


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