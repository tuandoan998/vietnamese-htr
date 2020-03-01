import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from skimage.feature import hog
from skimage import exposure
import re
import editdistance as ed
from collections import defaultdict, Counter
import glob
import io

class ScaleImageByHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        new_width = int(width * factor)
        new_height = int(height * factor)
        image = image.resize((new_width, new_height))
        return image
    
class HandcraftFeature(object):
    def __init__(self, orientations=8):
        self.orientations = orientations
    
    def __call__(self, image):
        image_width, image_height = image.size
        handcraft_img = np.ndarray((image_height, image_width, 3))
        handcraft_img[:, :, 0] = np.array(image.convert('L'))
        handcraft_img[:, :, 1] = np.array(image.filter(ImageFilter.FIND_EDGES).convert('L'))
        _, hog_image = hog(image, orientations=self.orientations, visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        handcraft_img[:, :, 2] = np.array(hog_image_rescaled)
        handcraft_img = Image.fromarray(np.uint8(handcraft_img))
        return handcraft_img
    
class PaddingWidth(object):
    '''
    Padding width in case of the width is too small
    '''
    def __init__(self, min_width):
        self.min_width = min_width

    def __call__(self, image):
        image_width, image_height = image.size
        width = self.min_width if image_width < self.min_width else image_width
        padded_image = Image.new('RGB', (width, image_height), color='white')
        padded_image.paste(image)
        return padded_image

def accuracy(outputs, targets):
    batch_size = outputs.size(0)
    _, ind = outputs.topk(1, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
    levenshtein distance
    tf-3gram
    letter n-grams
TO-DO:
    various corpus
    confuse character: a, á, à, â
    word-bigram probabilities.
'''
class Spell():
    def __init__(self, corpus_folder='data/corpus'):
        self.corpus_folder = corpus_folder
        self.corpus_words = self._corpus_words()
        self.dict_words = [word for word in self.corpus_words]
        self.build_language_model()
    
    def _words(self, text):
        return re.findall(r'\w+', text.lower())
    
    def _corpus_words(self):
        print('Loading corpus ...')
        corpus_words = Counter()
        for file in glob.glob(self.corpus_folder+"/*/*.txt"):
            with open(file, encoding='utf-16', errors='ignore') as f:
                try:
                    data = f.read()
                    corpus_words += Counter(self._words(data))
                except:
                    continue
        print('Done!')
        corpus_words = {k:corpus_words[k] for k in corpus_words if corpus_words[k] > 1} # ignore word with freq = 1
        return corpus_words
    
    def _trigrams(self, word, padding=True):
        trigrams = []
        if padding:
            trigrams += [(None, None, word[0]), (word[-1], None, None)]
            if len(word) > 1:
                trigrams += [(None, word[0], word[1]), (word[-2], word[-1], None)]
        trigrams += [(word[i], word[i+1], word[i+2]) for i in range(len(word)-2)]
        return trigrams
    
    def build_language_model(self): ## different with thesis
        self.lm = defaultdict(lambda: defaultdict(lambda: 0))
        for word in self.corpus_words:
            for c1, c2, c3 in self._trigrams(word):
                self.lm[(c1, c2)][c3] += self.corpus_words[word]
        for c1_c2 in self.lm:
            total_count = float(sum(self.lm[c1_c2].values()))
            for c3 in self.lm[c1_c2]:
                self.lm[c1_c2][c3] /= total_count
    
    def correction(self, predict_words):
        res = []
        for word in predict_words:
            word = ''.join(word)
            if word.lower() in self.dict_words or word.replace('.','',1).isdigit(): # hypothesis
                res.append([c for c in word])
            else:
                candidates = self._levenshtein_candidates(word)
                score = dict()
                for candidate in candidates:
                    score.update({candidate: self._tf_3gram(candidate, word) + self._n_gram_score(candidate)})
                candidate_word = max(score.keys(), key=lambda k: score[k])
                
                if candidate_word.islower() and word[0].isupper(): # hypothesis
                    candidate_word = candidate_word.replace(candidate_word[0], candidate_word[0].upper(), 1)
                if ed.distance(candidate_word, word.lower()) >= len(word)*2/3: # hypothesis
                    candidate_word = word
                res.append([c for c in candidate_word])
        return res
                
    def _levenshtein_candidates(self, predict_word):
        candidates = list()
        dist = dict()
        for word in self.dict_words:
            dist.update({word: ed.distance(predict_word, word)})
        min_dist = min(dist.items(), key=lambda x: x[1])[1]
        for key, value in dist.items():
            if value == min_dist:
                candidates.append(key)
        return candidates
    
    # scoring matches using a simple Term Frequency (TF) count
    def _tf_3gram(self, word1, word2):
        tf_count = 0
        word1 = '##'+word1+'##'
        word2 = '##'+word2+'##'
        n_grams1 = [word1[i:i+3] for i in range(len(word1)-2)]
        n_grams2 = [word2[i:i+3] for i in range(len(word2)-2)]
        for n_gram1 in n_grams1:
            for n_gram2 in n_grams2:
                if n_gram1==n_gram2:
                    tf_count += 1
                    break
        if len(word1)==len(word2): # hypothesis
            tf_count += 2
        return tf_count
    
    def _n_gram_score(self, word): ## different with thesis
        score = 1.0
        for c1, c2, c3 in self._trigrams(word):
            score *= self.lm[c1, c2][c3]
        score = score**(1/float(len(word)+2))
        return score
    
        
                
if __name__=='__main__':
    spell = Spell()
    print(len(spell.dict_words))
    print(spell.correction(['hượng', 'tồn', 'mai', 'kím']))