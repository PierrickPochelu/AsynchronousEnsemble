## Asynchronous ensembles

Different simple asynchronous ensemble designs are compared.


Given `n` processes (predicting on CIFAR10 images), `m` devices (CPU, GPU, or TPU), and the given assignment map `n[i]->m[j]`. We evaluate different design strategies: 
* Strategy "Sequential". It consists of iteratively (for-loop) each base model one by one, then their predictions are averaged.
* Strategy "Multiprocessing". It consists of Multiprocessing is a built-in python package to run multiple processes in parallel. Here, we launch multiple TF sessions in parallel for each base model.
* Strategy "1 TF session". It consists in combining base models in one single Tensorflow session.


ML tasks may interfere with each other negatively and experience performance unpredictability. This is due the parallel tasks share resources: core utilization, core caches, IO, buses.

<!--
Some more general papers [[1]](#1) [[2]](#2) exist to efficiently assign `n` heterogeneous AI models on `m` devices at inference time but do not provide code. They focus on the combinatorial optimization problem. Here, the assignment map `n[i]->m[j]` is given. Another axis of works consists in compiling/optimizing neural networks but not shown here.

<a id="1">[1]</a>  Automatic assignement based on RL: https://i2.cs.hku.hk/~cwu/papers/yxbao-infocom19.pdf

<a id="2">[2]</a>  Automatic assignement and tuning (batch size, data-parallel) processes:  https://arxiv.org/pdf/2208.14049.pdf
-->

## Common settings


```python
# Common imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, models
from tensorflow.keras import Input, Model, optimizers, layers
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")

# Common constants
NB_TRAINING_SAMPLES = 6000  # Number training samples for faster experiences
NB_INFERENCE_SAMPLES = 6000  # Number inference samples
ENSEMBLE_SIZE = 4
NB_EPOCHS = 1
GPUID = [0, 0, 0, 0]  #  # asignement of models with GPU ID or -1
BATCH_SIZE = 4

# Common base model
def keras_model(x):
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(10, activation="softmax")(x)
    x = layers.Softmax()(x)
    return x

# Common evaluation procedure
def evaluate_model(model, x, y):
    start_time = time.time()
    y_pred = model.predict(x, batch_size=BATCH_SIZE) # <--- inference is here
    enlapsed = time.time() - start_time
    acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
    return {"accuracy": round(acc, 2), "time": round(enlapsed, 2)}
  
# Common data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_images = train_images[:NB_TRAINING_SAMPLES]
train_labels = train_labels[:NB_TRAINING_SAMPLES]
test_images = test_images[:NB_INFERENCE_SAMPLES]
test_labels = test_labels[:NB_INFERENCE_SAMPLES]
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170498071/170498071 [==============================] - 2s 0us/step


##Strategy "Sequential"


```python
def keras_model_builder(config):
    input_shape = (32, 32, 3)
    loss = "categorical_crossentropy"
    opt = "adam"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpuid"])
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    input = Input(shape=input_shape)
    output = keras_model(input)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=loss, optimizer=opt)
    return model
def gen_pred(models, x, batch_size):
  for model_i in models:
    yield model_i.predict(x, batch_size=batch_size)

class ensemble:
    def __init__(self, ensemble_size, gpus):
        self.loss = "categorical_crossentropy"
        self.opt = "adam"
        self.ensemble_size=ensemble_size
        self.gpus=gpus
        self.models=[]
        for i in range(ensemble_size):
            config_i={"gpuid":self.gpus[i]}
            model_i=keras_model_builder(config_i)
            model_i.compile(loss=self.loss, optimizer=self.opt)
            self.models.append(model_i)

    def fit(self, train_images, train_labels):
        for model_i in self.models:
            model_i.fit(
                x=train_images, 
                y=train_labels, 
                batch_size=BATCH_SIZE, 
                epochs=NB_EPOCHS)

    def predict(self, x, batch_size):
        cumulated_preds=None
        g=gen_pred(self.models, x, batch_size)
        for preds_i in g:
          if cumulated_preds is None:
            cumulated_preds=preds_i
          else:
            cumulated_preds+=preds_i
        return cumulated_preds

# Training
ensemble = ensemble(ensemble_size=ENSEMBLE_SIZE, gpus=GPUID)
ensemble.fit(train_images, train_labels)

# Inference
for i, base_model in enumerate(ensemble.models):
    info = evaluate_model(base_model, test_images, test_labels)
    print(f"Model id: {i} accuracy: {info['accuracy']} time: {info['time']}")

info = evaluate_model(ensemble, test_images, test_labels)
print(f"Ensemble accuracy: {info['accuracy']} inference time: {info['time']}")

# Destroy
del ensemble
tf.keras.backend.clear_session()
```

    1500/1500 [==============================] - 19s 11ms/step - loss: 2.2526
    1500/1500 [==============================] - 18s 11ms/step - loss: 2.2707
    1500/1500 [==============================] - 19s 12ms/step - loss: 2.3430
    1500/1500 [==============================] - 17s 11ms/step - loss: 2.2494
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 0 accuracy: 0.24 time: 6.34
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 1 accuracy: 0.2 time: 6.43
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 2 accuracy: 0.1 time: 6.84
    1500/1500 [==============================] - 7s 4ms/step
    Model id: 3 accuracy: 0.26 time: 10.41
    1500/1500 [==============================] - 8s 5ms/step
    1500/1500 [==============================] - 6s 4ms/step
    1500/1500 [==============================] - 6s 4ms/step
    1500/1500 [==============================] - 6s 4ms/step
    Ensemble accuracy: 0.24 inference time: 31.84


##Strategy "Multiprocessing"


```python
from multiprocessing import Queue, Process
import math
SEGSIZE=500

def keras_model_builder(config):
    input_shape = (32, 32, 3)
    loss = "categorical_crossentropy"
    opt = "adam"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpuid"])
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    input = Input(shape=input_shape)
    output = keras_model(input)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=loss, optimizer=opt)
    return model

class MyProcess(Process):
    def __init__(
        self,
        rank: int,
        config: dict,
        dataset: list,
        shared_input_queue: Queue,
        shared_output_queue: Queue,
    ): 
        # WARNING: don't use global variable in MyProcess 
        # otherwise -> unexpected behaviour and deadlock may happen
        Process.__init__(self, name="ModelProcessor")
        self.rank = rank
        self.config = config
        self.dataset = dataset  # List of np.ndarray xtrain, ytrain, xtest, ytest
        self.shared_input_queue = shared_input_queue  # 'go' or 'stop'
        self.shared_output_queue = shared_output_queue  # "initok" or predictions
        self.model = None
        self.info = None

    def _asynchronous_predict(self):
        finish = False
        def generator():
          x=self.dataset[2]
          for segi in range(0,len(x),self.config["SEGSIZE"]):
            segment_i=x[i:i+self.config["SEGSIZE"]]
            segout = self.model.predict(segment_i, 
                                        batch_size=BATCH_SIZE, 
                                        verbose=0)
            yield segi, segout
        gen=generator()
        while finish == False:
            msg = self.shared_input_queue.get()  # wait
            if msg == "go":
                for segi, segout in gen:
                  self.shared_output_queue.put((self.rank, segi, segout))
            else:
                finish = True

    def train(self):
        self.model.fit(
            x=self.dataset[0],
            y=self.dataset[1],
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
        )

    def run(self):
        self.model = keras_model_builder(self.config)
        self.train()
        info = evaluate_model(self.model, x=self.dataset[2], y=self.dataset[3])
        self.shared_output_queue.put((self.rank, -1, info))  # notify the main process

        self._asynchronous_predict()  # run forever

############
# TRAINING #
############
input_queue = Queue()
output_queue = Queue()
processes = []


# Launch training process
for i in range(ENSEMBLE_SIZE):
    proc = MyProcess(
        rank=i,
        config={"gpuid": GPUID[i], "epochs": NB_EPOCHS, 
                "batch_size": BATCH_SIZE, "SEGSIZE": SEGSIZE},
        dataset=[train_images, train_labels, test_images, test_labels],
        shared_input_queue=input_queue,
        shared_output_queue=output_queue,
    )
    proc.start()  # start and wait
    processes.append(proc)
print("The building/training is launched ...")

# Wait every processes is ready
training_start_time = time.time()
for i in range(ENSEMBLE_SIZE):
    thread_id, seg_id, msg = output_queue.get()
    if isinstance(msg, dict):
        print(
            f"Model rank: {thread_id} accuracy: {msg['accuracy']} time: {msg['time']}"
        )
    else:
        raise ValueError(f"thread {thread_id} received an unexpected message")

#############
# Inference #
#############
def gen_read(processes, output_queue):
    nbreq=math.ceil(float(len(test_images))/SEGSIZE) * len(processes)
    for i in range(nbreq):
      thread_id, segi, segout = output_queue.get()
      yield segi, segout
preds = np.zeros(test_labels.shape, np.float32)
gen=gen_read(processes, output_queue)

inference_start_time = time.time()
for process in processes:
    input_queue.put("go")

for segi, segout in gen:
    start=segi
    end=min( segi+len(segout) , len(preds) )
    preds[start:end] = segout + preds[start:end]

inf_time = time.time() - inference_start_time
acc = np.mean(np.argmax(test_labels, axis=1) == np.argmax(preds, axis=1))
print(f"Ensemble accuracy: {round(acc,2)} inference time: {round(inf_time,2)}")

# Stop processes
for i in range(ENSEMBLE_SIZE):
    input_queue.put("stop")
del processes
tf.keras.backend.clear_session()

```

    The building/training is launched ...
    1500/1500 [==============================] - 76s 49ms/step - loss: 2.2539
    1500/1500 [==============================] - 77s 49ms/step - loss: 2.2543
    1500/1500 [==============================] - 78s 49ms/step - loss: 2.2505
    1500/1500 [==============================] - 78s 50ms/step - loss: 2.2588
    1500/1500 [==============================] - 30s 20ms/step
    1500/1500 [==============================] - 29s 19ms/step
    1500/1500 [==============================] - 30s 19ms/step
    1500/1500 [==============================] - 30s 20ms/step
    Model rank: 2 accuracy: 0.28 time: 32.43
    Model rank: 1 accuracy: 0.23 time: 42.42
    Model rank: 0 accuracy: 0.1 time: 41.94
    Model rank: 3 accuracy: 0.1 time: 42.05
    Ensemble accuracy: 0.1 inference time: 21.72


##Strategy "1 TF Session"
.


```python
def from_gpu_id_to_device_name(gpuid):
    if gpuid == -1:
        return "/device:CPU:0"
    else:
        return "/device:GPU:" + str(gpuid)


class ensemble:
    def __init__(self, ensemble_size, gpus):
        self.loss = "categorical_crossentropy"
        self.opt = "adam"

        self.model_list = []
        output_list = []

        with tf.device(from_gpu_id_to_device_name(gpus[0])):
            input = Input(shape=(32, 32, 3))

        for i in range(ensemble_size):
            with tf.device(from_gpu_id_to_device_name(gpus[i])):
                input_i = tf.identity(input)
                output_i = keras_model(input_i)
                model_i = Model(inputs=input_i, outputs=output_i)
                output_list.append(output_i)
                self.model_list.append(model_i)

        with tf.device(from_gpu_id_to_device_name(gpus[0])):
            merge = tf.stack(output_list, axis=-1)
            combined_predictions = tf.reduce_mean(merge, axis=-1)

        self.ensemble = Model(inputs=input, outputs=combined_predictions)
        self.ensemble.compile(loss=self.loss, optimizer=self.opt)

    def fit(self, train_images, train_labels):
        for model_i in self.model_list:
            model_i.compile(loss=self.loss, optimizer=self.opt)
            model_i.fit(
                x=train_images, y=train_labels, 
                batch_size=BATCH_SIZE, epochs=NB_EPOCHS)

    def predict(self, x, batch_size):
        return self.ensemble.predict(x, batch_size=batch_size)

# Training
ensemble = ensemble(ensemble_size=ENSEMBLE_SIZE, gpus=GPUID)
ensemble.fit(train_images, train_labels)

# Inference
for i, base_model in enumerate(ensemble.model_list):
    info = evaluate_model(base_model, test_images, test_labels)
    print(f"Model id: {i} accuracy: {info['accuracy']} time: {info['time']}")

info = evaluate_model(ensemble, test_images, test_labels)
print(f"Ensemble accuracy: {info['accuracy']} inference time: {info['time']}")

# Destroy
del ensemble
tf.keras.backend.clear_session()
```

    1500/1500 [==============================] - 16s 11ms/step - loss: 2.2504
    1500/1500 [==============================] - 16s 11ms/step - loss: 2.2625
    1500/1500 [==============================] - 17s 11ms/step - loss: 2.2632
    1500/1500 [==============================] - 17s 11ms/step - loss: 2.3295
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 0 accuracy: 0.25 time: 6.81
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 1 accuracy: 0.1 time: 6.51
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 2 accuracy: 0.21 time: 10.43
    1500/1500 [==============================] - 6s 4ms/step
    Model id: 3 accuracy: 0.1 time: 10.38
    1500/1500 [==============================] - 15s 10ms/step
    Ensemble accuracy: 0.19 inference time: 15.02


## Experimental results

Results obtained with Google Colab.
 
| Device | Iterative design | Multiprocessing design | 1 TF Session design |
| ------ | ------ | ------ | ------ |
| GPU | 16.33 sec | 9.68 sec | 4.65 sec |
| TPU | 36.37 sec | 25.98 sec | 20.78 sec |
| CPU | 31.84 sec | 21.72 sec | 15.02 sec |

## Conclusion 
The asynchronous ensemble designs exploits the underlying parallelism. 

Multiprocessing design requires each process know the data. Thus, memory consumption for data and buses communication are multiplied. In addition of that, 1 Tensorflow session may optimize more efficiently the assignment of tensors and cores than independant sessions. However, multiple independant process may allow more flexibility for ML applications, like predicting with different inference frameworks (e.g., Torch, Tensorflow, JAX, ...).

We may expect the results vary according models size, hardware, ... MIG technology (multi-instances) may offer better performance and deterministic performance.


```python

```
