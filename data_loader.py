import pickle
import gc
import copy
import json
import random
from glob import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Data_cleaner():
    def __init__(self, config, type):
        self.config = config
        self.type = type
        self.raw_data = []
        self.num_article = len(self.raw_data)

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model'])
        self.MAX_TITLE_LEN = config['MAX_TITLE_LEN']
        self.MAX_TAG_LEN = config['MAX_TAG_LEN']

    # load raw data
    def load(self):
        print('Data_cleaner:\tStart loading dataset')
        for year in tqdm(self.config[self.type + '_year']):
            for file in glob('./data/*/data/'+year+'/*.json'):
                with open(file) as f:
                    raw_data = json.load(f)
                if type(raw_data['data']) is list and len(raw_data['data'])>0:
                    self.raw_data += raw_data['data']
                    print('Data_cleaner:\tLoading raw dataset: {}'.format(file))
        self.num_article = len(self.raw_data)
        print('Data_cleaner:\tFinished loading dataset')
        print('Data_cleaner:\t# article: {}'.format(self.num_article))

    def _cut_or_pad(self, MAX_LEN, ids):
        # cutting
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        # padding
        elif len(ids) < MAX_LEN:
            num_pad = MAX_LEN - len(ids)
            ids += [0 for _ in range(num_pad)]

        return ids

    # tokenize raw data
    def tokenize(self):
        print('Data_cleaner:\tStart tokenizing articles')
        for article in tqdm(self.raw_data):
            article['title_tokens'] = self.tokenizer.tokenize(article['title'])
            article['tags_tokens'] = (self.tokenizer.tokenize(
                tag) for tag in article['tags'] if type(tag) is str)

            article['title'] = ''.join(article['title_tokens'])
            article['tags'] = (''.join(tag_tokens)
                               for tag_tokens in article['tags_tokens'])

            article['title_ids'] = self.tokenizer.convert_tokens_to_ids(
                article['title_tokens'])
            article['tags_ids'] = (self.tokenizer.convert_tokens_to_ids(
                tag_tokens) for tag_tokens in article['tags_tokens'])

            article['title_ids_trim'] = self._cut_or_pad(
                self.MAX_TITLE_LEN, article['title_ids'])
            article['tags_ids_trim'] = [self._cut_or_pad(
                self.MAX_TAG_LEN, id_tokens) for id_tokens in article['tags_ids']]

            # del article['title_tokens'], article['tags_tokens'], article['title'], article['tags'], article['title_ids'], article['tags_ids']
            # gc.collect()

        print('Data_cleaner:\tFinished tokenizing articles')


class Data_manager():
    def __init__(self, config, data):
        self.config = config
        self.data = data
        if self.config['shuffle']:
            random.shuffle(self.data)
        self.num_article = len(data)
        self.pos_indexed_tokens = []
        self.neg_indexed_tokens = []
        self.neg_Y = []

    # positive example
    def create_positive_idx_tokens(self):
        print('Data_manager:\tStart Creating Positive Tokens...')
        self.pos_indexed_tokens = [article['title_ids_trim'] +
                                   tag_ids for article in self.data for tag_ids in article['tags_ids_trim']]
        print('Data_manager:\tDone Creating Positive Tokens')

    # negative example
    # should be random each time
    def create_negative_idx_tokens(self):
        print('Data_manager:\tStart Creating Negative Tokens...')
        for article in self.data:
            for _ in article['tags_ids_trim']:
                candidate = article['tags_ids_trim'][0]
                while(candidate in article['tags_ids_trim']):
                    rand_idx = random.randrange(self.num_article)
                    if len(self.data[rand_idx]['tags_ids_trim'])>0:
                        candidate = random.choice(self.data[rand_idx]['tags_ids_trim'])

                assert candidate not in article['tags_ids_trim']

                self.neg_indexed_tokens.append(
                    article['title_ids_trim'] + candidate)

        print('Data_manager:\tDone Creating Negative Tokens')

    # |DATA| x MAX_LEN x EBD_DIM
    def make_epoch(self):
        print('Data_manager:\tStart Making Epoch...')
        # Convert inputs to PyTorch tensors
        self.tokens_tensor = torch.tensor(
            self.pos_indexed_tokens + self.neg_indexed_tokens, dtype=torch.long)
        self.segments_tensors = torch.tensor([[0] * self.config['MAX_TITLE_LEN'] +
                                              [1] * self.config['MAX_TAG_LEN'] for _ in range(self.tokens_tensor.shape[0])], dtype=torch.long)

        self.label = torch.tensor([1] * len(self.pos_indexed_tokens) +
                                  [0] * len(self.neg_indexed_tokens), dtype=torch.float).view(-1)

        print('Data_manager:\tDone Making Epoch')

    def get_fitting_features_labels(self):
        print('Data_manager:\tStart Fitting features...')
        if len(self.pos_indexed_tokens) == 0:
            self.create_positive_idx_tokens()
        if self.config['type'] != 'predict':
            self.create_negative_idx_tokens()
        self.make_epoch()

        print('Data_manager:\t# Total Pairs: {}\t#Pos Pairs: {}\t#Neg Pairs: {}'.format(self.label.size(0), len(self.pos_indexed_tokens), len(self.neg_indexed_tokens)))

        assert self.label.size(0) == self.tokens_tensor.size(0)
        assert self.label.size(0) == self.segments_tensors.size(0)

        return self.tokens_tensor, self.segments_tensors, self.label


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config, tokens, seg, label):
        'Initialization'
        self.tokens = tokens
        self.seg = seg
        self.labels = label.detach()

        self.config = config

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.tokens)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = torch.cat((self.tokens[index].view(
            1, -1), self.seg[index].view(1, -1)), 1)
        y = self.labels[index]

        return X, y
