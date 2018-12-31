from comet_ml import Experiment # import comet_ml FIRST!!
import keras

# create an experiment with your api key
experiment = Experiment(api_key="YOUR_API_KEY",
                        project_name='YOUR_PROJECT_NAME')

# hypyer-params: should be a DICTIONARY
params={'batch_size':batch_size,
        'epochs':epochs,
        'layer1_type':'Dense',
        'layer1_num_nodes':num_nodes,
        'layer1_activation':activation,
        'optimizer':optimizer
}
experiment.log_parameters(params)

# will log metrics with the prefix 'train_'
with experiment.train():
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-4,patience=3, verbose=1, mode='auto')])

# will log metrics with the prefix 'test_'
with experiment.test():
    loss, accuracy = model.evaluate(x_test, y_test)
    metrics = {
        'loss':loss,
        'accuracy':accuracy
    }
    experiment.log_metrics(metrics)