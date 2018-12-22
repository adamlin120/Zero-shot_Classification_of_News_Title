import apex
import multiprocessing
import datetime
from pytorch_pretrained_bert import BertModel
from pprint import pprint
import copy
from ast import literal_eval
import numpy as np
import torch
from torch.utils import data
from data_loader import Data_cleaner, Data_manager, Dataset

# load config
with open('./configs/config_example.txt', 'r') as f:
    config = literal_eval(f.read())
    config['type'] = 'predict'
    config['shuffle'] = False
    config['predict_raw_file_list'] = ['/home/adam/Desktop/irtm/final_project/requests/sample_request.txt']
    # pprint(config)

# load model
model = torch.load(config['model_for_predict'])
print(model)

# load input
raw_data = []
with open(config['predict_raw_file_list'][0], 'r', encoding='utf-8') as f:
    for line in f.readlines():
        req_id, title, tags = line.strip().split('|||')
        req_id = req_id.strip()
        title = title.strip()
        tags = list(map(lambda x: x.strip(), tags.strip().split()))
        
        raw_data += [{'title': title, 'tags': tags, 'req_id':req_id, 'num_tags': len(tags)}]

print(raw_data)

data_cleaner = Data_cleaner(config, 'predict')
data_cleaner.raw_data = copy.deepcopy(raw_data)
data_cleaner.num_article = len(data_cleaner.raw_data)
data_cleaner.tokenize()

data_manager = Data_manager(config, data_cleaner.raw_data)
tokens_tensor, segments_tensors, label = data_manager.get_fitting_features_labels()

dataset = Dataset(config, tokens_tensor, segments_tensors, label)
generator = data.DataLoader(
    dataset, batch_size=config['val_batch_size'], num_workers=multiprocessing.cpu_count(), pin_memory=True)


# BERT init
bert_model = BertModel.from_pretrained(config['bert_model'])
BERT_LAYERS = config['BERT_LAYERS']
BERT_INTER_LAYER = config['BERT_INTER_LAYER']

# check devices
if torch.cuda.is_available():
    device = torch.device('cuda')
    # move model to GPU
    model.cuda()
    bert_model.cuda()
else:
    device = torch.device('cpu')
print("SYSTEM:\tUsing Device ", device)

# Evaluation
model.eval()
result = np.array([], dtype=np.float)
with torch.set_grad_enabled(False):
    for local_batch, local_labels in (generator):
        # convert to BERT feature
        local_batch = local_batch.squeeze()
        tokens_tensor, segments_tensors = local_batch[:, :(
            config['MAX_TITLE_LEN']+config['MAX_TAG_LEN'])], local_batch[:, (config['MAX_TITLE_LEN']+config['MAX_TAG_LEN']):]
        tokens_tensor, segments_tensors = tokens_tensor.to(
            device), segments_tensors.to(device)
        bert_model.eval()
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
        encoded_layers = encoded_layers[(-BERT_LAYERS):]
        local_batch = torch.zeros_like(encoded_layers[0])
        if BERT_INTER_LAYER == 'mean':
            for i in range(BERT_LAYERS):
                # local_batch += encoded_layers[i]/BERT_LAYERS
                local_batch += encoded_layers[i]
        elif BERT_INTER_LAYER == 'concat':
            local_batch = torch.cat(encoded_layers, 2)

        # fuse features
        if config['model'] == 'simpleBiLinear':
            local_batch = torch.cat((torch.mean(local_batch[:, :config['MAX_TITLE_LEN'], :], (1)).squeeze(
            ), torch.mean(local_batch[:, config['MAX_TITLE_LEN']:, :], (1)).squeeze()), 1)

        # Transfer to GPU
        local_batch = local_batch.to(device)
        # forward pass
        y_pred = model(local_batch)
        # cal metrics
        result = np.concatenate([result, y_pred.data.cpu().numpy().reshape((-1,))])
print(result)

ret = {'result': [], 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
cur = 0
for data in raw_data:
    ret['result'] += [{'req_id': data['req_id'], 'score': list(result[cur:(cur+data['num_tags'])]), 'title': data['title'], 'tags': data['tags'], 'num_tags': data['num_tags']}]
    cur += data['num_tags']


pprint(ret)
