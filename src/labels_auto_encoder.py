import keras
import argparse
import h5py
#import json_file
import sys
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.models import load_model
#from keras.models import model_from_json


def build_label_autoencoder(name, noise_input=0.2, size_layers=[20, 10],
        dropout_layers=[0.25, 0.25], n_code=8, epochs=50, batch_size=100):
    """
    Builds an autoencoder on the labels
    """

    # load the train data
    with h5py.File('../data/processed/y_train.h5', 'r') as hf:
           y_train = hf['y_train'][:]

    n_train, n_labels = y_train.shape
    train_indices = list(range(n_train))


    # inputs
    inputs = Input((n_labels,), name='input')
    encoded = Dropout(rate=noise_input)(inputs)

    # encoding
    for n_hidden, dropout_rate in zip(size_layers, dropout_layers):
        encoded = Dense(units=n_hidden, activation='relu')(encoded)
        encoded = Dropout(rate=dropout_rate)(encoded)

    encoded = Dense(units=n_code, activation='sigmoid', name='encoding')(encoded)

    # decoding
    first_layer = True
    for n_hidden, dropout_rate in zip(reversed(size_layers),
                                                    reversed(dropout_layers)):
        if first_layer:
            decoded = Dense(units=n_hidden, activation='relu',
            name='decoding')(encoded) 
            first_layer = False
        else:
            decoded = Dense(units=n_hidden, activation='relu')(decoded)
        decoded = Dropout(rate=dropout_rate)(decoded)

    decoded = Dense(units=n_labels, activation='sigmoid', name='out')(decoded)

    autoencoder = Model(inputs=inputs, outputs=decoded)
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.fit(x=y_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
    autoencoder.save('../models/{}_autoencoder.h5'.format(name))
    encoder = Model(inputs=inputs, outputs=encoded)
    encoder.save('../models/{}_encoder.h5'.format(name))
    
    # building decoder
    encoded_input = Input(shape=(n_code,))
    
    ### NOT WORKING FOR DROPOUT LAYERS != 2 ###
    d1 = autoencoder.layers[-5]
    d2 = autoencoder.layers[-4]
    d3 = autoencoder.layers[-3]
    d4 = autoencoder.layers[-2]
    d5 = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=d5(d4(d3(d2(d1(encoded_input))))))
    decoder.save('../models/{}_decoder.h5'.format(name))

    #with open('../models/{}_architecture.json'.format(name), 'w') as json_file:
    #    json_file.write(model.to_json())

def load_autoencoder(name):
    """
    Loads a pretrained autoencoder and returns the encoder and decoder
    """
    autoencoder = load_model('../models/{}_autoencoder.h5'.format(name))
    encoder = load_model('../models/{}_encoder.h5'.format(name))
    decoder = load_model('../models/{}_decoder.h5'.format(name))

    return autoencoder, encoder, decoder
    #model = model_from_json('../models/{}_architecture.json'.format(name))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('name', type=str, help="name of your model")
    parser.add_argument('-n','--noise_input', type=float, default=0.2, help="fraction of inputs to discard during training")
    parser.add_argument('-l','--size_layers', type=int, nargs='+', default=[20, 10], help='list of sizes of hidden layers (symmetic encoding/decoding)')
    parser.add_argument('-d','--dropout_layers', type=float, nargs='+', default=[0.25, 0.25], help='dropout applied to hidden layers (symmetic encoding/decoding)')
    parser.add_argument('-c','--n_code', type=int, default=8, help='dimension of the encoding')
    parser.add_argument('-e', '--epochs', type=int, default=50, help="number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=100, help="batch size")

    args = parser.parse_args()

    build_label_autoencoder(**vars(args))

if __name__ == "__main__":
    sys.exit(main())
