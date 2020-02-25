import argparse
import datetime
import os

import editdistance as ed
import numpy as np
import torch
import tqdm
from torchvision import transforms

from data import EOS_CHAR, get_data_loader, get_vocab
from model import DenseNetFE, Seq2Seq, Transformer, LeNetFE
from utils import ScaleImageByHeight, HandcraftFeature


def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    
    print('Load weight from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    config = checkpoint['config']

    cnn_config = config['densenet']
    if config['common']['cnn'] == 'densenet':
        cnn_config = config['densenet']
        cnn = DenseNetFE(cnn_config['depth'],
                         cnn_config['n_blocks'],
                         cnn_config['growth_rate'])
    elif config['common']['cnn'] == 'lenet':
        cnn = LeNetFE()
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))
    vocab = get_vocab(config['common']['dataset'], config['common']['cnn'])

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

    if config['common']['cnn'] == 'densenet':
        test_transform = transforms.Compose([
            ScaleImageByHeight(config['common']['scale_height']),
            HandcraftFeature() if config['common']['use_handcraft'] else transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
    elif config['common']['cnn'] == 'lenet':
        test_transform = transforms.Compose([
            ScaleImageByHeight(config['common']['scale_height']),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    test_loader = get_data_loader(config['common']['dataset'], 'test', config['common']['batch_size'],
                                  config['common']['cnn'], test_transform, vocab, debug=args.debug)
    
    model.eval()
    CE = 0
    WE = 0
    total_characters = 0
    total_words = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            imgs, targets, targets_onehot, lengths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            # outputs, _ = model.greedy(imgs, targets_onehot[[0]].transpose(0,1))
            # _, index = outputs.topk(1, -1)
            # predicts = index.squeeze() # [B, T]
            predicts, _ = model.beamsearch(imgs, targets_onehot[[0]].transpose(0,1))
            
            predicts_str = []
            for predict in predicts:
                s = [vocab.int2char[x.item()] for x in predict]
                try:
                    eos_index = s.index(EOS_CHAR) + 1
                except ValueError:
                    eos_index = len(s)
                predicts_str.append(s[:eos_index-1])

            targets_str = []
            for target in targets.transpose(0, 1).squeeze():
                s = [vocab.int2char[x.item()] for x in target]
                try:
                    eos_index = s.index(EOS_CHAR) + 1
                except ValueError:
                    eos_index = len(s)
                targets_str.append(s[1:eos_index-1])
            
            assert len(predicts_str) == len(targets_str)
            for j in range(len(predicts_str)):
                CE += ed.distance(predicts_str[j], targets_str[j])
            total_characters += (lengths-2).sum().item()
            
            for j in range(len(predicts_str)):
                if not np.array_equal(np.array(predicts_str[j]), np.array(targets_str[j])):
                    WE += 1
            total_words += len(predicts_str)

    CER = CE / total_characters
    WER = WE / total_words
    print(f'Predict wrong {CE}/{total_characters}. CER={CER}')
    print(f'Predict wrong {WE}/{total_words}. WER={WER}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    main(args)