import pickle
import gc
import json
import random
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
        for file in tqdm(self.config[self.type+'_raw_file_list']):
            print('Data_cleaner:\tLoading raw dataset: {}'.format(file))
            with open(file) as f:
                raw_data = json.load(f)
            self.raw_data += raw_data['data']
        self.num_article = len(self.raw_data)
        print('Data_cleaner:\tFinished loading dataset')
        print('Data_cleaner:\t# article: {}'.format(self.num_article))

    def _cut_or_pad(self, MAX_LEN, ids):
        # cutting
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        # padding
        elif len(ids) < MAX_LEN:
            num_pad = MAX_LEN-len(ids)
            ids += [0 for _ in range(num_pad)]

        return ids

    # tokenize raw data
    def tokenize(self):
        print('Data_cleaner:\tStart tokenizing articles')
        for article in tqdm(self.raw_data):
            article['title_tokens'] = self.tokenizer.tokenize(article['title'])
            article['tags_tokens'] = [
                self.tokenizer.tokenize(tag) for tag in article['tags']]

            article['title'] = ''.join(article['title_tokens'])
            article['tags'] = [''.join(tag_tokens)
                               for tag_tokens in article['tags_tokens']]

            article['title_ids'] = self.tokenizer.convert_tokens_to_ids(
                article['title_tokens'])
            article['tags_ids'] = [self.tokenizer.convert_tokens_to_ids(
                tag_tokens) for tag_tokens in article['tags_tokens']]

            article['title_ids_trim'] = self._cut_or_pad(
                self.MAX_TITLE_LEN, article['title_ids'])
            article['tags_ids_trim'] = [self._cut_or_pad(
                self.MAX_TAG_LEN, id_tokens) for id_tokens in article['tags_ids']]

            del article['title_tokens'], article['tags_tokens'], article['title'], article['tags'], article['title_ids'], article['tags_ids']
            gc.collect()

        print('Data_cleaner:\tFinished tokenizing articles')


class Data_manager():
    def __init__(self, config, data):
        self.config = config
        self.data = data
        if self.config['shuffle']:
            random.shuffle(self.data)
        self.num_article = len(data)
        self.pos_indexed_tokens = []

        # self.bert_model = BertModel.from_pretrained(config['bert_model'])
        # self.BERT_LAYERS = config['BERT_LAYERS']
        # self.BERT_INTER_LAYER = config['BERT_INTER_LAYER']

    # positive example
    def create_positive_idx_tokens(self):
        self.pos_indexed_tokens = [article['title_ids_trim'] +
                                   tag_ids for article in self.data for tag_ids in article['tags_ids_trim']]
        self.pos_Y = [
            1 for article in self.data for tag_ids in article['tags_ids_trim']]

    # negative example
    # should be random each time
    def create_negative_idx_tokens(self):
        self.neg_indexed_tokens = []
        self.neg_Y = [
            0 for article in self.data for tag_ids in article['tags_ids_trim']]
        for article in self.data:
            for _ in article['tags_ids_trim']:
                candidate = random.choice(
                    self.data[random.randrange(self.num_article)]['tags_ids_trim'])
                while(candidate in article['tags_ids_trim']):
                    candidate = random.choice(
                        self.data[random.randrange(self.num_article)]['tags_ids_trim'])

                assert candidate not in article['tags_ids_trim']

                self.neg_indexed_tokens.append(
                    article['title_ids_trim'] + candidate)

    # def extract_bert_feature(self, bert_model, tokens_tensor, segments_tensors):
    #     bert_model.eval()

    #     # Predict hidden states features for each layer
    #     encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
    #     encoded_layers = encoded_layers[(-self.BERT_LAYERS):]

    #     return encoded_layers

    # |DATA| x MAX_LEN x EBD_DIM
    def make_epoch(self):
        # Convert inputs to PyTorch tensors
        self.tokens_tensor = torch.tensor(
            self.pos_indexed_tokens + self.neg_indexed_tokens, dtype=torch.long)
        self.segments_tensors = torch.tensor([[0]*self.config['MAX_TITLE_LEN'] + [
            1]*self.config['MAX_TAG_LEN'] for _ in range(self.tokens_tensor.shape[0])], dtype=torch.long)

        # self.encoded_layers = self.extract_bert_feature(
        #     self.bert_model, self.tokens_tensor, self.segments_tensors)

        # self.features = torch.zeros_like(self.encoded_layers[0])
        # if self.BERT_INTER_LAYER == 'mean':
        #     for i in range(self.BERT_LAYERS):
        #         self.features += self.encoded_layers[i]/self.BERT_LAYERS
        # elif self.BERT_INTER_LAYER == 'concat':
        #     self.features = torch.cat(self.encoded_layers, 2)

        # # fuse features
        # if self.config['model']=='simpleBiLinear':
        #     self.features = torch.mean(self.features, (1))

        self.label = torch.tensor(
            self.pos_Y + self.neg_Y, dtype=torch.float).view(-1)

        # # momery release
        # del self.encoded_layers
        # gc.collect()


    def get_fitting_features_labels(self):
        if len(self.pos_indexed_tokens) == 0:
            self.create_positive_idx_tokens()
        self.create_negative_idx_tokens()
        self.make_epoch()

        # self.features = self.features.detach()
        # return self.features, self.label

        return self.tokens_tensor, self.segments_tensors, self.label


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config, tokens, seg, label):
        'Initialization'
        self.tokens = tokens
        self.seg = seg
        self.labels = label.detach()

        self.config = config
    #     self.bert_model = BertModel.from_pretrained(config['bert_model'])
    #     self.BERT_LAYERS = config['BERT_LAYERS']
    #     self.BERT_INTER_LAYER = config['BERT_INTER_LAYER']

    # def extract_bert_feature(self, tokens_tensor, segments_tensors):
    #     self.bert_model.eval()

    #     # Predict hidden states features for each layer
    #     encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)
    #     encoded_layers = encoded_layers[(-self.BERT_LAYERS):]

    #     return encoded_layers

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.tokens)

    def __getitem__(self, index):
        'Generates one sample of data'
        # # Load data and get label
        # encoded_layers = self.extract_bert_feature(
        #     self.tokens[index].view(1, -1), self.seg[index].view(1, -1))

        # X = torch.zeros_like(encoded_layers[0])
        # if self.BERT_INTER_LAYER == 'mean':
        #     for i in range(self.BERT_LAYERS):
        #         X += encoded_layers[i]/self.BERT_LAYERS
        # elif self.BERT_INTER_LAYER == 'concat':
        #     X = torch.cat(encoded_layers, 2)

        # # fuse features
        # if self.config['model'] == 'simpleBiLinear':
        #     X = torch.mean(X, (1)).squeeze()

        # X = X.detach()
        X = torch.cat((self.tokens[index].view(1, -1), self.seg[index].view(1, -1)), 1)
        y = self.labels[index]

        return X, y
