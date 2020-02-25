import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from .attention import get_attention

class Decoder(nn.Module):
    def __init__(self, attention, feature_size, hidden_size, vocab_size, attn_size):
        super(Decoder, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_size = attn_size

        self.rnn = nn.LSTM(
            input_size=self.vocab_size+self.feature_size,
            hidden_size=self.hidden_size,
        )

        self.attention = get_attention(attention, feature_size, hidden_size, attn_size)

        self.character_distribution = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, img_features, targets, teacher_forcing_ratio=0.5):
        '''
        :param img_features: tensor of [num_pixels, B, C]
        :param targets: tensor of [T, B, V], each target has <start> at beginning of the word
        :return:
            outputs: tensor of [T, B, V]
            weights: tensor of [T, B, num_pixels]
        '''
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)
        max_length = targets.size(0)

        targets = targets.float()
        rnn_input = targets[[0]].float() # [1, B, V]
        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)

        for t in range(max_length):
            context, weight = self.attention(hidden, img_features) # [1, B, C], [num_pixels, B, 1]
            self.rnn.flatten_parameters()
            output, (hidden, cell_state) = self.rnn(torch.cat((rnn_input, context), -1), (hidden, cell_state))
            output = self.character_distribution(output)

            outputs[[t]] = output

            teacher_force = random.random() < teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = targets[[t]]
            else:
                rnn_input = output

        return outputs

    def greedy(self, img_features, start_input, max_length=10):
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)

        rnn_input = start_input.float()

        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)
        
        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device) 

        # pdb.set_trace()
        for t in range(max_length):
            context, weight = self.attention(hidden, img_features, output_weights=True) # [B, 1, num_pixels]

            rnn_input = torch.cat((rnn_input, context), -1)
            output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))
            output = self.character_distribution(output)

            outputs[[t]] = output
            weights[[t]] = weight.transpose(0, 1)

            rnn_input = output

        return outputs, weights
    
    def beamsearch(self, img_features, start_input, max_length=10, beam_size=3):
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)
        rnn_input = start_input.float()
        rnn_inputs = [rnn_input] * beam_size

        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)
        hiddens = [hidden] * beam_size
        cell_states = [cell_state] * beam_size

        outputs = torch.zeros(max_length, batch_size, beam_size, device=img_features.device, dtype=torch.long)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device)
        beam_weights = [weights] * 3

        sequences = [[list(), 1.0]]
        for t in range(max_length):
            for i in range(beam_size):
                context, weight = self.attention(hiddens[i], img_features, output_weights=True) # [B, 1, num_pixels]
                rnn_output, (hiddens[i], cell_states[i]) = self.rnn(torch.cat((rnn_inputs[i], context), -1), (hiddens[i], cell_states[i]))
                rnn_output = self.character_distribution(rnn_output)
                rnn_inputs[i] = rnn_output
                beam_weights[i][[t]] = weight.transpose(0, 1)
                
                rnn_output = F.softmax(rnn_output, dim=2).squeeze() # [B, V]
                outputs[[t], :] = rnn_output.topk(beam_size, -1)[1]
                print(outputs)
                return

def beam_search_decoder(data, k=3):
    res = []
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -torch.log(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]
        
        res += [x for x in sequences if x[0][-1] == char2int[EOS_CHAR]]
        sequences = [x for x in sequences if x[0][-1] != char2int[EOS_CHAR]]
        if len(res)==3:
            break
    if len(res)==0:
        return sequences[0][0]
    return sorted(res, key=lambda tup:tup[1])[0][0]