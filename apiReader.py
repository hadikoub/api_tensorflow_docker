from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    test_acc: float
    test_loss: float


class Param(BaseModel):
    epochs: int
    optimizer: str
    loss_function: str
    traning_perc: float
    dataset_folder: str


def search(param: Param):

    dataset="dataset/"+str(param.dataset_folder)
    df = pd.read_csv(dataset)
    labels = df['label']
    images = df[df.columns[1:785]]
    images = np.asarray(images)
    images = images.reshape(len(images), 28, 28)
    train_labels, test_labels, train_images, test_images = train_test_split(labels, images,
                                                                            test_size=(1 - param.traning_perc))
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.0

    test_images = test_images / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=param.optimizer,
                  loss=param.loss_function,
                  metrics=['accuracy'])
    # batches= int(len(train_images)/param.batches)
    l0 = model.fit(train_images, train_labels, epochs=param.epochs )
    # test the accuracy with test model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    model.save('mymodel.h5')
    result = [test_loss, test_acc]

    return result


@app.get("/")
def read_root():
    return {"test_acc": "100",
            "test_loss": "0"}


@app.put("/get_output/")
async def item(param: Param):
    result = search(param)
    return {"test_acc": str(result[1]), "test_loss": str(result[0])}
