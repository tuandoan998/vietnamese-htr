from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import os
import tornado.wsgi
import tornado.httpserver

import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from data import get_data_loader, get_vocab, EOS_CHAR, SOS_CHAR
from model import DenseNetFE, Seq2Seq, Transformer
from utils import ScaleImageByHeight, HandcraftFeature
from metrics import CharacterErrorRate, WordErrorRate

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from word_segmentation import wordSegmentation, prepareImg
import cv2
import shutil

app = Flask(__name__)
REPO_DIRNAME = os.path.dirname(os.path.abspath(__file__))


class HTR(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device = {}'.format(self.device))
        checkpoint = torch.load(self.args.weight, map_location=self.device)
        config = checkpoint['config']

        if config['common']['cnn'] == 'densenet':
            cnn_config = config['densenet']
            cnn = DenseNetFE(cnn_config['depth'],
                             cnn_config['n_blocks'],
                             cnn_config['growth_rate'])

        self.vocab = get_vocab(config['common']['dataset'])

        model_config = config['tf']
        self.model = Transformer(cnn, self.vocab.vocab_size, model_config)
        self.model.load_state_dict(checkpoint['model'])

        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.test_transform = transforms.Compose([
            ScaleImageByHeight(config['common']['scale_height']),
            HandcraftFeature() if config['common']['use_handcraft'] else transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def word_predict(self, path):
        self.model.eval()        
        image = Image.open(path)
        image = self.test_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        sos_input = torch.nn.functional.one_hot(torch.tensor(self.vocab.char2int[SOS_CHAR]), num_classes=self.vocab.vocab_size)
        sos_input = sos_input.unsqueeze(0).unsqueeze(0) # [B,1,V] where B = 1
        sos_input = sos_input.to(self.device)
        outputs, weights = self.model.module.greedy(image, sos_input, output_weights=self.args.output_weights)
        outputs = outputs.topk(1, -1)[1] # [B,T,1]
        outputs = outputs.to('cpu')
        outputs = outputs.squeeze(0).squeeze(-1).tolist() # remove batch and 1
        outputs = [self.vocab.int2char[output] for output in outputs]
        outputs = ''.join(outputs[:outputs.index(EOS_CHAR)])
        return outputs

    def run(self, path):
        try:
            with graph.as_default():
                output = []
                img = prepareImg(cv2.imread(path), 64)
                ws = wordSegmentation(img, kernelSize=25, sigma=11, theta=4, minArea=200)
                if not os.path.exists('tmp'):
                    os.mkdir('tmp')
                for (j, w) in enumerate(ws):
                    (word_bbox, word_img) = w
                    cv2.imwrite('tmp/%d.png'%j, word_img)
                img_files = os.listdir('tmp')
                img_files = sorted(img_files)
                for f in img_files:
                    text = self.word_predict('tmp/'+f)
                    print(text)
                    output.append(text)
                shutil.rmtree('tmp')
            return output

        except Exception as err:
            print('Prediction error: ', err)
            return (False, 'Something went wrong when predict the '
                           'image. Maybe try another one?')


@app.route("/")
def index():
    return render_template("template.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(REPO_DIRNAME, 'tmp_images')
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    output = htr_model.run(destination)
    return render_template("template.html", predict=output, image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("tmp_images", filename)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--output-weights', action='store_true', default=False)
    parser.add_argument('--beamsearch', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    global htr_model
    htr_model = HTR(args)
    global graph
    graph = tf.get_default_graph()

    start_tornado(app)
