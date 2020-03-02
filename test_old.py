import argparse
import datetime
import os

import editdistance as ed
import numpy as np
import torch
import tqdm
from torchvision import transforms

from data import EOS_CHAR, get_data_loader, get_vocab
from model import DenseNetFE, Seq2Seq, Transformer
from utils import ScaleImageByHeight, HandcraftFeature, Spell, PaddingWidth


def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    
    print('Load weight from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    config = checkpoint['config']

    cnn_config = config['densenet']
    cnn = DenseNetFE(cnn_config['depth'],
                     cnn_config['n_blocks'],
                     cnn_config['growth_rate'])
    vocab = get_vocab(config['common']['dataset'])

    if args.model == 'tf':
        model_config = config['tf']
        model = Transformer(cnn, vocab.vocab_size, model_config)
    elif args.model == 's2s':
        model_config = config['s2s']
        model = Seq2Seq(cnn, vocab.vocab_size, model_config['hidden_size'], model_config['attn_size'])
    else:
        raise ValueError('model should be "tf" or "s2s"')
    model.to(device)

    model.load_state_dict(checkpoint['model'])

    test_transform = transforms.Compose([
        ScaleImageByHeight(config['common']['scale_height']),
        HandcraftFeature() if config['common']['use_handcraft'] else transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_loader = get_data_loader(config['common']['dataset'], 'test', config['common']['batch_size'],
                                  test_transform, vocab, debug=args.debug)
    
    non_word = open('./non_word.txt', 'w+')
    real_word_same = open('./real_word_same.txt', 'w+')
    real_word_diff = open('./real_word_diff.txt', 'w+')
    print(f'{"G-truth": <{10}} | {"Predict": <{10}} | {"Edited": <{10}} | {"Image path": <{65}}', file=non_word)
    print(f'{"G-truth": <{10}} | {"Predict": <{10}} | {"Edited": <{10}} | {"Image path": <{65}}', file=real_word_same)
    print(f'{"G-truth": <{10}} | {"Predict": <{10}} | {"Edited": <{10}} | {"Image path": <{65}}', file=real_word_diff)
    spell = Spell()
    model.eval()
    CE = 0
    WE = 0
    total_characters = 0
    total_words = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            imgs, targets, targets_onehot, lengths, paths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            outputs, _ = model.greedy(imgs, targets_onehot[[0]].transpose(0,1))

            _, index = outputs.topk(1, -1)
            predicts = index.squeeze(-1) # [B, T]
            
            predicts_str = []
            for predict in predicts:
                s = [vocab.int2char[x.item()] for x in predict]
                try:
                    eos_index = s.index(EOS_CHAR) + 1
                except ValueError:
                    eos_index = len(s)
                predicts_str.append(s[:eos_index-1])

            edits_str = spell.correction(predicts_str)
            
            if args.edit_predicts == True:
                compares_str = edits_str
            else:
                compares_str = predicts_str

            targets_str = []
            for target in targets.transpose(0, 1).squeeze(-1):
                s = [vocab.int2char[x.item()] for x in target]
                try:
                    eos_index = s.index(EOS_CHAR) + 1
                except ValueError:
                    eos_index = len(s)
                targets_str.append(s[1:eos_index-1])
            
            assert len(predicts_str) == len(targets_str)
            for j in range(len(targets_str)):
                CE += ed.distance(compares_str[j], targets_str[j])
            total_characters += (lengths-2).sum().item()
            
            for j in range(len(targets_str)):
                if not np.array_equal(np.array(compares_str[j]), np.array(targets_str[j])):
                    WE += 1
                    if np.array_equal(np.array(edits_str[j]), np.array(predicts_str[j])):
                        if ed.distance(''.join(predicts_str[j]).lower(), ''.join(targets_str[j]).lower()) <= 1:
                            print(f'{"".join(targets_str[j]): <{10}} | {"".join(predicts_str[j]): <{10}} | {"".join(edits_str[j]): <{10}} | {paths[j]: <{65}}', file=real_word_same)
                            real_word_same.flush()
                        else:
                            print(f'{"".join(targets_str[j]): <{10}} | {"".join(predicts_str[j]): <{10}} | {"".join(edits_str[j]): <{10}} | {paths[j]: <{65}}', file=real_word_diff)
                            real_word_diff.flush()
                    else:
                        print(f'{"".join(targets_str[j]): <{10}} | {"".join(predicts_str[j]): <{10}} | {"".join(edits_str[j]): <{10}} | {paths[j]: <{65}}', file=non_word)
                        non_word.flush()
            total_words += len(targets_str)

    CER = CE / total_characters
    WER = WE / total_words
    print(f'Predict wrong {CE}/{total_characters}. CER={CER}')
    print(f'Predict wrong {WE}/{total_words}. WER={WER}')
    
    print(f'Predict wrong {CE}/{total_characters}. CER={CER}', file=non_word)
    print(f'Predict wrong {WE}/{total_words}. WER={WER}', file=non_word)
    print(f'Predict wrong {CE}/{total_characters}. CER={CER}', file=real_word_same)
    print(f'Predict wrong {WE}/{total_words}. WER={WER}', file=real_word_same)
    print(f'Predict wrong {CE}/{total_characters}. CER={CER}', file=real_word_diff)
    print(f'Predict wrong {WE}/{total_words}. WER={WER}', file=real_word_diff)
    non_word.close()
    real_word_same.close()
    real_word_diff.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--edit-predicts', action='store_true', default=False)
    args = parser.parse_args()

    main(args)