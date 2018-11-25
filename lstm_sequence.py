from keras.layers import LSTM
from keras.models import Sequential
from Unit import SpindleData
from keras.layers import Flatten, Dense, Embedding
import numpy as np
import os
from Unit import draw
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))      #将GPU设置为按需分配


def learning_lstm():                   #lstm暂时还是比较适合于文本中，对于有序序暂不合适
    x_train, y_labels, length = data_test()
    x_train = np.expand_dims(x_train, axis=2)

    model = Sequential()
    # model.add(Embedding(max_feature, 32))
    model.add(LSTM(64, input_shape=(length, 1)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    print("DIMs=%d"%(x_train[0].shape[0]))
    history = model.fit(x_train, y_labels, epochs=10, batch_size=32, validation_split=0.2)
    draw(history)


def data_test():
    length = 0   #每一个系列的长度
    spindle = SpindleData(step=0.001)
    x_train = spindle.coding_q
    y_train = spindle.labels
    length = spindle.max_length
    return x_train, y_train, length


learning_lstm()
