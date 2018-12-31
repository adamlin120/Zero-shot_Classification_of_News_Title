from comet_ml import Experiment
import os
from pytorch_pretrained_bert import BertModel
import gc
from pprint import pprint
import multiprocessing
from ast import literal_eval
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from data_loader import Data_cleaner, Data_manager, Dataset
from net import simpleBiLinear, attention
from util import count_parameters, get_config_sha1

# load config
with open('./configs/config_linear.txt', 'r') as f:
    config = literal_eval(f.read())
    config['config_sha1'] = get_config_sha1(config, 5)
    pprint(config)

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="fdb4jkVkz4zT8vtOYIRIb0XG7",
                        project_name="irtm-final-project", workspace="adamlin120",
                        auto_output_logging=None)
experiment.log_parameters(config)

# load raw data
train_data_cleaner = Data_cleaner(config, 'train')
train_data_cleaner.load()
train_data_cleaner.tokenize()
val_data_cleaner = Data_cleaner(config, 'val')
val_data_cleaner.load()
val_data_cleaner.tokenize()

# init data manager
train_data_manager = Data_manager(config, train_data_cleaner.raw_data)
train_tokens_tensor, train_segments_tensors, train_label = train_data_manager.get_fitting_features_labels()
val_data_manager = Data_manager(config, val_data_cleaner.raw_data)
val_tokens_tensor, val_segments_tensors, val_label = val_data_manager.get_fitting_features_labels()

# Generators
training_set = Dataset(config, train_tokens_tensor,
                       train_segments_tensors, train_label)
training_generator = data.DataLoader(training_set, batch_size=config['batch_size'], num_workers=multiprocessing.cpu_count(
), shuffle=config['shuffle'], pin_memory=True)
val_set = Dataset(config, val_tokens_tensor, val_segments_tensors, val_label)
val_generator = data.DataLoader(
    val_set, batch_size=config['val_batch_size'], num_workers=multiprocessing.cpu_count(), pin_memory=True)

# init model, optim, criterion, scheduler
model = eval(config['model'])(config)
opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    opt, factor=config['lr_sche_factor'], patience=config['lr_sche_patience'], verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 3, 0.5)
criterion = nn.BCELoss()
print(model)
print("Number of Trainable Parameters: ", count_parameters(model))

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


# metric
global_step = 0
metric = {'train_loss': [], 'train_accu': [], 'val_loss': [], 'val_accu': []}

# Loop over epochs
for epoch in trange(config['epochs'], desc='1st loop'):
    scheduler.step()
    # Training
    with experiment.train():
        model.train()
        for i, (local_batch, local_labels) in tqdm(enumerate(training_generator), desc='2nd loop', leave=False):
            global_step += 1
            # convert to BERT feature
            local_batch = local_batch.squeeze()
            tokens_tensor, segments_tensors = local_batch[:, :(
                config['MAX_TITLE_LEN'] + config['MAX_TAG_LEN'])], local_batch[:, (config['MAX_TITLE_LEN'] + config['MAX_TAG_LEN']):]
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

            # Transfer to GPU / (CPU)
            local_batch, local_labels = local_batch.to(
                device).squeeze(), local_labels.to(device).squeeze()

            # Model computations
            opt.zero_grad()
            # forward pass
            y_pred = model(local_batch)
            # cal loss
            loss = criterion(y_pred, local_labels)
            # backward pass
            loss.backward()
            # update
            opt.step()

            # cal metrics
            y_pred_class = (
                np.sign(y_pred.clone().detach().cpu().view(-1) - 0.5) + 1) / 2
            num_corrct = float((local_labels.clone().detach().cpu() == y_pred_class).to(torch.float).sum())
            accu = num_corrct / len(local_labels)
            metric['train_loss'].append(loss.item())
            metric['train_accu'].append(accu)
            # Log to Comet.ml
            experiment.log_metric("loss", loss.item(), step=global_step)
            experiment.log_metric("accuracy", accu, step=global_step)

            tqdm.write("\t\tTrain Accuracy: {}\tTrain Loss: {}".format(
                accu, loss.item()))

    # Validation
    with experiment.test():
        model.eval()
        with torch.set_grad_enabled(False):
            num_correct = 0
            num_pair = 0
            epoch_loss = 0
            for i, (local_batch, local_labels) in tqdm(enumerate(val_generator), desc='2nd loop', leave=False):
                # convert to BERT feature
                local_batch = local_batch.squeeze()
                tokens_tensor, segments_tensors = local_batch[:, :(
                    config['MAX_TITLE_LEN'] + config['MAX_TAG_LEN'])], local_batch[:, (config['MAX_TITLE_LEN'] + config['MAX_TAG_LEN']):]
                tokens_tensor, segments_tensors = tokens_tensor.to(
                    device), segments_tensors.to(device)
                bert_model.eval()
                encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
                encoded_layers = encoded_layers[(-BERT_LAYERS):]
                local_batch = torch.zeros_like(encoded_layers[0])
                if BERT_INTER_LAYER == 'mean':
                    for i in range(BERT_LAYERS):
                        local_batch += encoded_layers[i]
                elif BERT_INTER_LAYER == 'concat':
                    local_batch = torch.cat(encoded_layers, 2)

                if config['model'] == 'simpleBiLinear':
                    # fuse features
                    local_batch = torch.cat((torch.mean(local_batch[:, :config['MAX_TITLE_LEN'], :], (1)).squeeze(
                    ), torch.mean(local_batch[:, config['MAX_TITLE_LEN']:, :], (1)).squeeze()), 1)
                # elif config['model'] == 'lstm':
                #     #

                # Transfer to GPU / (CPU)
                local_batch, local_labels = local_batch.to(
                    device).squeeze(), local_labels.to(device).squeeze()
                # forward pass
                y_pred = model(local_batch)
                # cal metrics
                loss = criterion(y_pred, local_labels)
                y_pred_class = (
                    (np.sign(y_pred.detach().cpu().view(-1) - 0.5) + 1) / 2)
                num_correct += float((local_labels.detach().cpu() == y_pred_class).to(torch.float).sum())
                num_pair += len(local_labels)
                epoch_loss += loss.clone().detach() / len(local_labels)


            if epoch == 0 or epoch_loss < list(sorted(metric['val_loss']))[0]:
                torch.save(model, os.path.join(
                    config['model_save_dir'], 'model_{}_{}_{}.pt'.format(config['model'], config['config_sha1'], global_step)))

            accu = num_correct / num_pair
            metric['val_loss'].append(epoch_loss)
            metric['val_accu'].append(accu)

            experiment.log_metric("loss", epoch_loss, step=global_step)
            experiment.log_metric("accuracy", accu, step=global_step)

            tqdm.write("\t\tVal Accuracy: {}\tVal Loss: {}".format(
                accu, epoch_loss))
