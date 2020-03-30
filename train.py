import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.utils import setup_logger
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torchvision import transforms

from dataset import get_data_loader
from model import ModelTF, ModelRNN, DenseNetFE, SqueezeNetFE, EfficientNetFE, CustomFE, ResnetFE
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import logging
import hydra

# Reproducible
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = f'cuda' if torch.cuda.is_available() else 'cpu'

class OutputTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.tf = StringTransform(vocab, batch_first)

    def __call__(self, output):
        return list(map(self.tf, output[2:]))

@hydra.main(config_path='config/config.yaml', strict=False)
def main(cfg):
    logger = logging.getLogger('MainTraining')
    logger.info('Device = {}'.format(device))

    if cfg.get('resume', False):
        logger.info('Resuming from {}'.format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=device)
        cfg = checkpoint['config']
    best_metrics = dict()

    image_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(cfg.common['scale_height']),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_loader = get_data_loader(cfg.dataset['name'],
                                   cfg.common['train_partition'],
                                   cfg.common['batch_size'],
                                   cfg.common['num_workers'],
                                   image_transform,
                                   cfg.get('debug', False))

    val_loader = get_data_loader(cfg.dataset['name'],
                                 cfg.common['val_partition'],
                                 cfg.common['batch_size'],
                                 cfg.common['num_workers'],
                                 image_transform,
                                 cfg.get('debug', False))

    if cfg.get('debug', False) == True:
        vocab = train_loader.dataset.dataset.vocab
    else:
        vocab = train_loader.dataset.vocab
    logger.info('Vocab size = {}'.format(vocab.size))

    cnn = hydra.utils.instantiate(cfg.cnn)
    if cfg.decoder.name == 'transformer':
        model = ModelTF(cnn, vocab, cfg.decoder)
    elif cfg.decoder.name == 'rnn':
        model = ModelRNN(cnn, vocab, cfg.decoder)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    # multi_gpus = torch.cuda.device_count() > 1 and args.multi_gpus
    # if multi_gpus:
    #     logger.info("Let's use %d GPUs!", torch.cuda.device_count())
    #     model = nn.DataParallel(model, dim=0) # batch dim = 0

    if cfg.get('debug_model', False) == True:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        print(model)
        model.eval()
        dummy_image_input = torch.rand(cfg.common['batch_size'], 3, cfg.common['scale_height'], cfg.common['scale_height'] * 2)
        dummy_target_input = torch.rand(cfg.common['batch_size'], cfg.common['max_length'], vocab.size)
        dummy_output_train = model(dummy_image_input, dummy_target_input)
        dummy_output_greedy, _ = model.greedy(dummy_image_input, dummy_target_input[:,[0]])
        logger.debug(dummy_output_train.shape)
        logger.debug(dummy_output_greedy.shape)
        logger.info('Ok')
        exit(0)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss(gamma=2, alpha=vocab.class_weight).to(device)

    if cfg.optimizer.name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=cfg.optimizer['lr'],
                                  weight_decay=cfg.optimizer['weight_decay'],
                                  momentum=cfg.optimizer['momentum'])
    elif cfg.optimizer.name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.optimizer['lr'],
                               weight_decay=cfg.optimizer['weight_decay'])
    elif cfg.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.optimizer['lr'],
                              momentum=cfg.optimizer['momentum'],
                              weight_decay=cfg.optimizer['weight_decay'])
    else:
        raise ValueError(f'Unknow optimizer {cfg.optimizer.name}')
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=cfg.scheduler['n_epochs_decrease_lr'],
        min_lr=cfg.scheduler['end_lr'],
        verbose=True)

    if cfg.resume:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    log_dir = os.getcwd()
    tb_logger = TensorboardLogger(log_dir)
    CKPT_DIR = os.path.join(tb_logger.writer.get_logdir(), 'weights')
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    def step_train(engine, batch):
        model.train()
        optimizer.zero_grad()

        imgs, targets = batch.images.to(device), batch.labels.to(device)

        outputs = model(imgs, targets[:, :-1])

        outputs = pack_padded_sequence(outputs, (batch.lengths - 1), batch_first=True)[0]
        targets = pack_padded_sequence(targets[:, 1:], (batch.lengths - 1), batch_first=True)[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        return outputs, targets

    @torch.no_grad()
    def step_val(engine, batch):
        model.eval()

        imgs, targets = batch.images.to(device), batch.labels.to(device)
        logits = model(imgs, targets[:, :-1])
        outputs = model.greedy(imgs, targets[:, 0], cfg.common['max_length'])
        # if multi_gpus:
        #     outputs, _ = model.module.greedy(imgs, targets[:, [0]], output_weights=False)
        # else:
        #     outputs, _ = model.greedy(imgs, targets[:, [0]], output_weights=False)

        logits = pack_padded_sequence(logits, (batch.lengths - 1), batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets[:, 1:], (batch.lengths - 1), batch_first=True)[0]

        return logits, packed_targets, outputs, targets[:, 1:]

    trainer = Engine(step_train)
    Running(Loss(criterion), reset_interval=cfg.common['log_interval']).attach(trainer, 'Loss')
    Running(Accuracy(), reset_interval=cfg.common['log_interval']).attach(trainer, 'Accuracy')

    train_pbar = ProgressBar(ncols=0, ascii=True, position=0)
    train_pbar.attach(trainer, 'all')
    trainer.logger = setup_logger('Trainer')
    tb_logger.attach(trainer,
                     event_name=Events.ITERATION_COMPLETED,
                     log_handler=OutputHandler(tag='Train',
                                               metric_names=['Loss', 'Accuracy']))

    evaluator = Engine(step_val)
    Running(Loss(criterion, output_transform=lambda output: output[:2])).attach(evaluator, 'Loss')
    Running(CharacterErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'CER')
    Running(WordErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')

    eval_pbar = ProgressBar(ncols=0, ascii=True, position=0)
    eval_pbar.attach(evaluator, 'all')
    evaluator.logger = setup_logger('Evaluator')
    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='Validation',
                                               metric_names=['Loss', 'CER', 'WER'],
                                               global_step_transform=global_step_from_engine(trainer)),
                                               event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        state = evaluator.run(val_loader)
        lr_scheduler.step(state.metrics['Loss'])

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine: Engine):
        is_better = lr_scheduler.is_better(engine.state.metrics['Loss'], lr_scheduler.best)
        lr_scheduler.step(engine.state.metrics['Loss'])
        to_save = {
            'config': cfg,
            # 'model': model.state_dict() if not multi_gpus else model.module.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trainer': trainer.state_dict(),
        }

        torch.save(to_save, os.path.join(CKPT_DIR, f'weights_epoch={trainer.state.epoch}_loss={engine.state.metrics["Loss"]:.3f}.pt'))
        if is_better:
            torch.save(to_save, os.path.join(CKPT_DIR, 'BEST.pt'))
            best_metrics.update(engine.state.metrics)

    logger.info('='*60)
    logger.info(model)
    logger.info('='*60)
    logger.info(cfg.pretty())
    logger.info('='*60)
    logger.info('Start training..')
    if cfg.resume:
        trainer.load_state_dict(checkpoint['trainer'])
        trainer.run(train_loader, max_epochs=5 if cfg.debug else cfg.common['max_epochs'], seed=None)
    else:
        trainer.run(train_loader, max_epochs=5 if cfg.debug else cfg.common['max_epochs'], seed=seed)
    print(best_metrics)
    tb_logger.close()

if __name__ == '__main__':
    main()
