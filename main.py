from system import CESystem, CTCSystem
import logging
import argparse
from config import Config
from typing import Dict

def setup_train(args: Dict):
    config = Config(args['config_path'], **args)
    # TODO: override config
    if args['loss'] == 'ce':
        system = CESystem
    else:
        system = CTCSystem
    del args['loss']
    system().train(config, args['checkpoint'])

def setup_test(args: Dict):
    system = CESystem if args['loss'] == 'ce' else CTCSystem
    del args['loss']
    checkpoint = args['checkpoint']
    system().test(checkpoint, args['validation'], args['beam_width'])


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('loss', choices=['ctc', 'ce'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug-model', action='store_true', default=False)
    parser.add_argument('--log-root', type=str, default='./runs')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    train_parser.set_defaults(func=setup_train)
    train_parser.add_argument('config_path', type=str)
    train_parser.add_argument('--comment', type=str, default='')
    train_parser.add_argument('--trainval', action='store_true', default=False)
    train_parser.add_argument('-c', '--checkpoint', type=str)

    test_parser = subparser.add_parser('test')
    test_parser.set_defaults(func=setup_test)
    test_parser.add_argument('checkpoint', type=str)
    test_parser.add_argument('--beam-width', type=int, default=1)
    test_parser.add_argument('--validation', action='store_true', default=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger('PIL').setLevel(level=logging.INFO)
    func = args.func
    args = vars(args)
    del args['func']
    func(args)
