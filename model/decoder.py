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

            _, index = output.topk(1, -1) # _, [B, 1, 1]
            index = index.squeeze(-1) # [B, 1]
            outputs[:, [t]] = index
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
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)
        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell = self.init_hidden(batch_size).to(img_features.device)
        
        completed_words = [[] for i in range(batch_size)]
        prev_top_tokens = [[] for i in range(batch_size)]
        next_top_tokens = [[] for i in range(batch_size)]
        start_char = WordBeamSearch(hidden, cell, start_input, beam_size, EOS_index)
        
        for i in range(batch_size):
            prev_top_tokens[i].append(start_char)

        pdb.set_trace()
        for t in range(max_length):
            for i in range(len(prev_top_tokens[0])):
                tokens = [token[i] for token in prev_top_tokens] # [B]

                hidden = tokens.hidden
                cell = tokens.cell
                attn_hidden = self.Hc(hidden.transpose(0,1))
                context, _ = self.attention(attn_hidden, img_features) # [1, B, C], [num_pixels, B, 1]

                rnn_input = torch.cat((token.last_rnn_input, context), -1) # [1, B, V+C]
                output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) # [B, 1, hidden_size], _, _
                output = self.character_distribution(output) # [B, 1, V]
                rnn_input = output
                
                top_v, top_i = output.topk(beam_size, -1) # [B, 1, beam_size], [B, 1, beam_size]
                terms, tops = tokens.add_top_k(top_v, top_i, hidden, cell, rnn_input)

                for batch_index in range(batch_size):
                    completed_words[batch_index].extend(terms[batch_index])
                    next_top_tokens[batch_index].extend(tops[batch_index])

            for batch_index in range(batch_size):
                next_top_tokens[batch_index].sort(key=lambda s: s[1], reverse=True) # s[1] is avg_score of incomplete-word
                prev_top_tokens[batch_index] = next_top_tokens[batch_index][:beam_size]
                next_top_tokens[batch_index] = []

        for batch_index in range(batch_size):
            completed_words[batch_index] += [token.to_word_score() for token in prev_top_tokens[batch_index]]
            completed_words[batch_index].sort(key=lambda x: x[1], reverse=True)
            completed_words[batch_index] = completed_words[batch_index][0][0] # top1 avg_score, list word_idxes
            completed_words[batch_index] = completed_words[batch_index] + [EOS_index]*(max_length-len(completed_words[batch_index])) # padding

        return torch.tensor(completed_words)

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
            raise ValueError("Calculate average score of word, but got no character")
        return sum(self.word_scores) / len(self.word_scores)

    def add_top_k(self, top_values, top_indexs, current_hidden, current_cell, current_rnn_input):
        batch_size = top_values.size(0)
        top_values = torch.log(top_values)
        terminates, incomplete_words = [[] for i in range(batch_size)], [[] for i in range(batch_size)]
        
        for batch_index in range(batch_size):
            for beam_index in range(self.beam_size):
                if top_indexs[batch_index][0][beam_index] == self.EOS_index:
                    terminates[batch_index].append(([[idx.item()] for idx in self.word_idxes[batch_index]] + [self.EOS_index], self.avgScore())) 
                    continue
                idxes = self.word_idxes
                scores = self.word_scores
                idxes.append(top_indexs[batch_index][0][beam_index])
                scores.append(top_values[batch_index][0][beam_index])
                incomplete_words[batch_index].append(WordBeamSearch(current_hidden, current_cell, current_rnn_input, self.beam_size, self.EOS_index, idxes, scores))
        return terminates, incomplete_words

    def to_word_score(self):
        return (self.word_idxes, self.avg_score())