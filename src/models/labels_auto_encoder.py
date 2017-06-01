import keras
import argparse
import h5py
#import json_file
#import json
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
    layer_output = Dropout(rate=noise_input)(inputs)

    # encoding
    for n_hidden, dropout_rate in zip(size_layers, dropout_layers):
        layer_output = Dense(units=n_hidden, activation='relu')(layer_output)
        layer_output = Dropout(rate=dropout_rate)(layer_output)

    encoding = Dense(units=n_code, activation='sigmoid', name='encoding')(layer_output)
    layer_output = encoding

    # decoding
    first_layer = True
    for n_hidden, dropout_rate in zip(reversed(size_layers),
                                                    reversed(dropout_layers)):
        if first_layer:
            layer_output = Dense(units=n_hidden, activation='relu',
            name='decoding')(layer_output)
            first_layer = False
        else:
            layer_output = Dense(units=n_hidden, activation='relu')(layer_output)
        layer_output = Dropout(rate=dropout_rate)(layer_output)

    outputs = Dense(units=n_labels, activation='sigmoid')(layer_output)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=y_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                                    shuffle=True)
    model.save('../models/{}_weights.h5'.format(name))

    #with open('../models/{}_architecture.json'.format(name), 'w') as json_file:
    #    json_file.write(model.to_json())

def load_autoencoder(name):
    """
    Loads a pretrained autoencoder and returns the encoder and decoder
    """
    model = load_model('../models/{}_weights.h5'.format(name))
    encoder = Model(inputs=model.input,
                            outputs=model.get_layer('encoding').output)
    decoder = Model(inputs=model.get_layer('decoding').input,
                            outputs=model.output)
    return encoder, decoder
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
