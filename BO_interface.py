import numpy as np
import keras

from optimization import BlackBoxFunction
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
import random

def lzip(*args):
    """
    Zip, but returns zipped result as a list.
    """
    return list(zip(*args))

class NN(BlackBoxFunction):
    def __init__(self, id_n):
        self.dim = 2
        self.lengths = [1,1]
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.id_n = id_n
        np.random.seed(id_n)
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')
        self.X_train /= 255.0
        self.X_test /= 255.0

        self.nb_classes = 10
        self.Y_train = keras.utils.to_categorical(y_train, self.nb_classes)
        self.Y_test = keras.utils.to_categorical(y_test, self.nb_classes)
    
    def get_dim(self):
        return self.dim
    
    def randomize_training(self, N=25000):
        target = np.random.choice(self.X_train.shape[0], N)
        train = np.array(np.zeros(self.X_train.shape[0]), dtype=bool)
        train[target] = True
        self.X_train_sub = self.X_train[train,:] 
        self.Y_train_sub = self.Y_train[train]
	
    def do_evaluate(self, x):
        K.clear_session()
        self.randomize_training()
        learning_rate_scaled, dropout = x.flatten()
        #Training
        learning_rate = pow(10, -5*learning_rate_scaled-2)
        nb_epoch = 5
        batch_size = 100

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=self.X_train_sub.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=learning_rate, rho=dropout)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        # training
        history = model.fit(self.X_train_sub, self.Y_train_sub,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=0,
                            validation_data=(self.X_test, self.Y_test),
                            shuffle=True)

        loss, acc = model.evaluate(self.X_test, self.Y_test, verbose=0)
        return 10*(1-acc-0.4) #To scale it roughly between 0 and 1 with the current settings
