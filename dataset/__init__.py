import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from .vocab import CollateWrapper

from .iam import IAM
from .rimes import RIMES
from .vnondb import VNOnDB

from hydra.utils import to_absolute_path

def _get_dataset_partition_helper(dataset, partition, transform):
    if dataset not in ['vnondb', 'rimes', 'iam']:
        raise ValueError('Should be: ' + str(['vnondb', 'rimes', 'iam']))

    if partition not in ['train', 'test', 'val', 'trainval']:
        raise ValueError('Should be: ' + str(['train', 'test', 'val', 'trainval']))

    if dataset == 'vnondb':
        if partition == 'test':
            return VNOnDB(to_absolute_path('./data/VNOnDB/test_word'), to_absolute_path('./data/VNOnDB/test_word.csv'), transform)
        if partition == 'train':
            return VNOnDB(to_absolute_path('./data/VNOnDB/train_word'), to_absolute_path('./data/VNOnDB/train_word.csv'), transform)
        if partition == 'val':
            return VNOnDB(to_absolute_path('./data/VNOnDB/validation_word'), to_absolute_path('./data/VNOnDB/validation_word.csv'), transform)
        if partition == 'trainval':
            train = VNOnDB(to_absolute_path('./data/VNOnDB/train_word'), to_absolute_path('./data/VNOnDB/train_word.csv'), transform)
            val = VNOnDB(to_absolute_path('./data/VNOnDB/validation_word'), to_absolute_path('./data/VNOnDB/validation_word.csv'), transform)
            return ConcatDataset([train, val])
        return None
    elif dataset == 'rimes':
        if partition == 'test':
            return RIMES(to_absolute_path('./data/RIMES/data_test'), to_absolute_path('./data/RIMES/grount_truth_test_icdar2011.txt'), transform)
        if partition == 'train':
            return RIMES(to_absolute_path('./data/RIMES/trainingsnippets_icdar/training_WR'), to_absolute_path('./data/RIMES/groundtruth_training_icdar2011.txt'), transform)
        if partition == 'val':
            return RIMES(to_absolute_path('./data/RIMES/validationsnippets_icdar/testdataset_ICDAR'), to_absolute_path('./data/RIMES/ground_truth_validation_icdar2011.txt'), transform)
        return None
    elif dataset == 'iam':
        if partition == 'test':
            return IAM(to_absolute_path('./data/IAM/splits/test.uttlist'), transform)
        if partition == 'train':
            return IAM(to_absolute_path('./data/IAM/splits/train.uttlist'), transform)
        if partition == 'val':
            return IAM(to_absolute_path('./data/IAM/splits/validation.uttlist'), transform)
        return None

    return None

def collate_fn(batch):
    return CollateWrapper(batch)

def get_data_loader(dataset, partition, batch_size, num_workers=1, transform=None, debug=False):
    data = _get_dataset_partition_helper(dataset, partition, transform)
    shuffle = partition == 'train'

    if debug:
        data = Subset(data, torch.arange(batch_size*5 + batch_size//2).numpy())
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)
    return loader
