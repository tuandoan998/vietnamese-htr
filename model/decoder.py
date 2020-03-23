import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from .attention import get_attention
import pdb

class Decoder(nn.Module):
    def __init__(self, attention, feature_size, hidden_size, vocab_size, attn_size):
        super(Decoder, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_size = attn_size

        self.rnn = nn.LSTM(
            input_size=self.vocab_size+self.attn_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.Hc = nn.Linear(hidden_size, attn_size)
        self.Ic = nn.Linear(feature_size, attn_size)
        self.attention = get_attention(attention, attn_size)

        self.character_distribution = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, img_features, targets, teacher_forcing_ratio=0.5):
        '''
        :param img_features: tensor of [B, num_pixels, C]
        :param targets: tensor of [B, T, V], each target has <start> at beginning of the word
        :return:
            outputs: tensor of [B, T, V]
            weights: tensor of [B, T, num_pixels]
        '''
        num_pixels = img_features.size(1)
        batch_size = img_features.size(0)
        max_length = targets.size(1)

        targets = targets.float()
        rnn_input = targets[:,[0]].float() # [B,1,V]

        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=img_features.device)
        attn_img = self.Ic(img_features) # [B, num_pixels, A]

        for t in range(max_length):
            attn_hidden = self.Hc(hidden.transpose(0,1)) # batch first for Linear
            context, weight = self.attention(attn_hidden, attn_img, attn_img) # [B, 1, attn_size], [B, 1, num_pixels]
            self.rnn.flatten_parameters()
            output, (hidden, cell_state) = self.rnn(torch.cat((rnn_input, context), -1), (hidden, cell_state))
            output = self.character_distribution(output)

            outputs[:, [t]] = output

            teacher_force = random.random() < teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = targets[:, [t]]
            else:
                rnn_input = output

        return outputs

    def greedy(self, img_features, start_input, max_length=10):
        '''
        Shapes:
        -------
        - img_features: [B,num_pixels,C]
        - start_input: [B,1,V]

        Returns:
        - outputs: [B,T]
        - weights: [B,T,num_pixels]
        '''
        num_pixels = img_features.size(1)
        batch_size = img_features.size(0)

        rnn_input = start_input.float()

        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(batch_size, max_length, device=img_features.device)
        weights = torch.zeros(batch_size, max_length, num_pixels, device=img_features.device)

        attn_img = self.Ic(img_features)

        for t in range(max_length):
            attn_hidden = self.Hc(hidden.transpose(0,1)) # forward Linear should be batch_first
            context, weight = self.attention(attn_hidden, attn_img, attn_img, output_weights=True) # [B, 1, num_pixels]

            rnn_input = torch.cat((rnn_input, context), -1)

            output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))
            output = self.character_distribution(output)
            rnn_input = output

            index = output.argmax(-1).squeeze() # [B]
            outputs[:, t] = index
            weights[:, [t]] = weight

        return outputs, weights

    def beamsearch(self, img_features, start_input, EOS_index, max_length=10, beam_size=3):
        '''
        Shapes:
        -------
        - img_features: [B,num_pixels,C]
        - start_input: [B,1,V]

        Returns:
        - outputs: [B,T]
        '''
        outputs = []
        batch_size = img_features.size(0)

        # pdb.set_trace()
        for batch_index in range(batch_size):
            hidden = self.init_hidden(1).to(img_features.device)
            cell = self.init_hidden(1).to(img_features.device)
            start_char = WordBeamSearch(hidden, cell, start_input[0].unsqueeze(0).float(), beam_size, EOS_index)
            attn_img = self.Ic(img_features[batch_index].unsqueeze(0))

            completed_words = []
            prev_top_tokens = []
            next_top_tokens = []
            prev_top_tokens.append(start_char)

            for t in range(max_length):
                for token in prev_top_tokens:

                    hidden = token.last_hidden
                    cell = token.last_cell
                    attn_hidden = self.Hc(hidden.transpose(0,1))
                    context, _ = self.attention(attn_hidden, attn_img, attn_img) # [B, 1, num_pixels], [num_pixels, B, 1]

                    rnn_input = torch.cat((token.last_rnn_input, context), -1) # [1, B, V+C]
                    output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) # [B, 1, hidden_size], _, _
                    output = self.character_distribution(output) # [B, 1, V]
                    rnn_input = output
                    
                    top_v, top_i = output.squeeze(1).topk(beam_size, -1) # [B, 1, beam_size], [B, 1, beam_size]
                    terms, tops = token.add_top_k(top_v, top_i, hidden, cell, rnn_input)

                    completed_words.extend(terms)
                    next_top_tokens.extend(tops)
                    

                next_top_tokens.sort(key=lambda s: s.avg_score(), reverse=True)
                prev_top_tokens = next_top_tokens[:beam_size]
                next_top_tokens = []

            completed_words.extend([token.to_word_score() for token in prev_top_tokens])
            completed_words.sort(key=lambda x: x[1], reverse=True)
            completed_words = completed_words[0][0] # top1 avg_score, list word_idxes
            completed_words = completed_words + [EOS_index]*(max_length-len(completed_words)) # padding

            outputs.append(completed_words)

        return torch.tensor(outputs)

class WordBeamSearch():
    def __init__(self, last_hidden, last_cell, last_rnn_input, beam_size, EOS_index, word_idxes=None, word_scores=None):
        self.last_hidden = last_hidden
        self.last_cell = last_cell
        self.last_rnn_input = last_rnn_input
        self.beam_size = beam_size
        self.EOS_index = EOS_index
        self.word_idxes = [] if word_idxes==None else word_idxes
        self.word_scores = [] if word_scores==None else word_scores

    def avg_score(self):
        if len(self.word_scores) == 0:
            # raise ValueError("Calculate average score of word, but got no character")
            return 0
        return sum(self.word_scores) / len(self.word_scores)

    def add_top_k(self, top_values, top_indexs, current_hidden, current_cell, current_rnn_input):
        top_values = torch.log(top_values)
        terminates, incomplete_words = [], []
        for beam_index in range(self.beam_size):
            if top_indexs[0][beam_index].item() == self.EOS_index:
                terminates.append(([idx for idx in self.word_idxes] + [self.EOS_index], self.avg_score())) 
                continue
            idxes = self.word_idxes[:]
            scores = self.word_scores[:]
            idxes.append(top_indexs[0][beam_index].item())
            scores.append(top_values[0][beam_index].item())
            incomplete_words.append(WordBeamSearch(current_hidden, current_cell, current_rnn_input, self.beam_size, self.EOS_index, idxes, scores))
        return terminates, incomplete_words

    def __str__(self):
        return ','.join(str(idx) for idx in self.word_idxes)

    def to_word_score(self):
        return ([idx for idx in self.word_idxes], self.avg_score())