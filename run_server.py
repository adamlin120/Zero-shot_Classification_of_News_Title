from threading import Thread
import numpy as np
import flask
import redis
import uuid
import time
import json
import sys
import io
import os
import multiprocessing
import datetime
import itertools
from pytorch_pretrained_bert import BertModel
from pprint import pprint
import copy
from ast import literal_eval
import numpy as np
import torch
from torch.utils import data as ptdata
from data_loader import Data_cleaner, Data_manager, Dataset


# initialize constants used for model inference
SERVER_PORT = 5010

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
REDIS_PORT = 6379
BATCH_SIZE = 32
SERVER_SLEEP = 0.1
CLIENT_SLEEP = 0.1

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
r = redis.Redis()
r.flushdb()
db = redis.StrictRedis(host="localhost", port=REDIS_PORT, db=0)
model = None

# load config
with open('./configs/config_example.txt', 'r') as f:
    config = literal_eval(f.read())
    config['type'] = 'predict'
    config['shuffle'] = False
    config['predict_raw_file_list'] = ['/home/adam/Desktop/irtm/final_project/requests/sample_request.txt']
    # pprint(config)

def classify_process(config):
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    print("* Loading model...")
    model_save_path = config['model_for_predict']
    model = torch.load(model_save_path)
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
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = []

        # loop over the queue
        for i, q in enumerate(queue):
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            # data = q['data']
            data = copy.deepcopy(q)['data']


            # check to see if the batch list is None
            if batch==[]:
                batch = [data]
            # otherwise, stack the data
            else:
                batch += [data]

            # update the list of image IDs
            imageIDs.append(q["id"])

        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            # classify the batch
            print("* Batch size: {}".format(len(batch)))

            data_cleaner = Data_cleaner(config, 'predict')
            data_cleaner.raw_data = copy.deepcopy(batch)
            data_cleaner.num_article = len(data_cleaner.raw_data)
            data_cleaner.tokenize()

            data_manager = Data_manager(config, data_cleaner.raw_data)
            tokens_tensor, segments_tensors, label = data_manager.get_fitting_features_labels()

            dataset = Dataset(config, tokens_tensor, segments_tensors, label)
            generator = ptdata.DataLoader(
                dataset, batch_size=config['val_batch_size'], num_workers=multiprocessing.cpu_count(), pin_memory=True)

            # Evaluation
            model.eval()
            bert_model.eval()
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


            ret = {'result': [], 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            cur = 0
            for data in batch:
                ret['result'] += [{'req_id': data['req_id'], 'score': list(result[cur:(cur+data['num_tags'])]), 'title': data['title'], 'tags': data['tags'], 'num_tags': data['num_tags']}]
                cur += data['num_tags']

            for (imageID, res) in zip(imageIDs, ret['result']):
                output = res
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

            # remove the set of images from our queue
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount
        time.sleep(SERVER_SLEEP)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        jason = flask.request.get_json()
        if jason is not None:
            # {'title': title, 'tags': tags, 'req_id':req_id, 'num_tags': len(tags)}
            jason['data']['tags'] = list(itertools.chain.from_iterable([x.split() for x in jason['data']['tags']]))
            jason['data']['num_tags'] = len(jason['data']['tags'])

            print('Incoming Request: ', jason)
            
            k = str(uuid.uuid4())
            d = copy.deepcopy(jason)
            d['id'] = k
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    output = json.loads(output)
                    data["title"] = output['title']
                    data['tags'] = output['tags']
                    data['req_id'] = output['req_id']
                    data['num_tags'] = output['num_tags']
                    data['score'] = output['score']
                    

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

            print(data)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print("* Starting model service...")
    t = Thread(target=classify_process, args=[config])
    t.daemon = True
    t.start()

    # start the web server
    print("* Starting web service...")
    app.run(host='0.0.0.0', port=SERVER_PORT)