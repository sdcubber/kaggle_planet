import keras
import argparse
import h5py
import json
import sys
from keras.models import Model
from keras.layers import Input, Dense, Dropout

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
    inputs = Input((n_labels,))
    layer_output = Dropout(rate=noise_input)(inputs)

    # encoding
    for n_hidden, dropout_rate in zip(size_layers, dropout_layers):
        layer_output = Dense(units=n_hidden, activation='relu')(layer_output)
        layer_output = Dropout(rate=dropout_rate)(layer_output)

    encoding = Dense(units=n_code, activation='sigmoid')(layer_output)
    layer_output = encoding

    # decoding
    for n_hidden, dropout_rate in zip(reversed(size_layers),
                                                    reversed(dropout_layers)):
        layer_output = Dense(units=n_hidden, activation='relu')(layer_output)
        layer_output = Dropout(rate=dropout_rate)(layer_output)

    outputs = Dense(units=n_labels, activation='sigmoid')(layer_output)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=y_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                                    shuffle=True)
    model.save('../models/{}_weights.h5'.format(name))
    with open('../models/{}_architecture.json'.format(name), 'w') as json_file:
        json_file.write(model.to_json())

def load_autoencoder(name):
    """
    Loads a pretrained autoencoder and returns the encoder and decoder
    """

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
    print(args)

    build_label_autoencoder(**vars(args))

if __name__ == "__main__":
    sys.exit(main())
