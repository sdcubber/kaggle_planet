# Keras model classes for the planet kaggle challenge
import numpy as np
import keras as k
import keras.backend as K
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, Concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def GFM_power_classifier(n_labels, features, predictions, unique_combinations):
    """GFM algorithm that uses estimates from the joint probability from a label power set classifier.
    See http://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf for details."""
    E_F = []
    optimal_predictions = []

    for instance in range(features.shape[0]):
        E = []
        h = []

        D = np.ndarray(shape=(n_labels, n_labels))

        for i in range(17):
            occurring_combinations = unique_combinations[unique_combinations[:,i] == 1,:]

            P_y = predictions[instance, unique_combinations[:,i] == 1]
            s_y = np.sum(occurring_combinations, axis=1)

            for k in range(17):
                D[i, k] = np.sum((2*P_y)/(s_y + (k+1)))


        for k in range(n_labels):
            # solve inner optimization
            h_k = np.zeros(n_labels)
            h_k[np.argsort(D[:,k])[::-1][:k+1]] = 1 # Set h_i=1 to k labels with highest delta_ik
            h.append(h_k)

            # store a value of ...
            E.append(np.dot(h_k,D[:,k]))

        # solve outer maximization problem
        h_F = h[np.argmax(E)]
        E_f = E[np.argmax(E)]

        # Return optimal predictor hF, E[F(Y, hF)]
        optimal_predictions.append(h_F)
        E_F.append(E_f)
    return(np.array(optimal_predictions), E_F)

class simple_autoencoder(object):
    def __init__(self, input_dim=17, encoding_dim=8):

        self.nbits = encoding_dim
        self.input_seq = Input(shape=(input_dim,))

        # "encoded" is the encoded representation of the input
        self.encoded = Dense(self.nbits, activation='sigmoid')(self.input_seq)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_dim, activation='sigmoid')(self.encoded)

        self.autoencoder = Model(self.input_seq, decoded)
        self.autoencoder.compile(loss='binary_crossentropy',
                           optimizer='adadelta', metrics=['accuracy'])

        self.trained=False

    def fit(self, input_train, input_val, n_epochs=100, verbose=0):
        self.history = self.autoencoder.fit(input_train, input_train,
                            epochs=n_epochs,
                            batch_size=200,
                            validation_data=(input_val, input_val), verbose=verbose)

        self.trained=True

    def encode(self, input_seq):
        assert self.trained, 'Autoencoder has not been trained yet.'
        encoder = Model(self.input_seq, self.encoded)
        return(encoder.predict(input_seq, verbose=1))


    def decode(self, input_code):
        assert self.trained, 'Autoencoder has not been trained yet.'
        encoded_input = Input(shape=(self.nbits,))
        decoder_layer = self.autoencoder.layers[-1]

        decoder = Model(encoded_input, decoder_layer(encoded_input))

        return(decoder.predict(input_code, verbose=1))

    def __str__(self):
        return """Simple autoencoder from the Keras blog. See https://blog.keras.io/building-autoencoders-in-keras.html"""



def DataAugmenter():
	"""Return ImageDataGenerator with specified settings"""
	generator = ImageDataGenerator(featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	rotation_range=90.,
	width_shift_range=0.10,
	height_shift_range=0.10,
	shear_range=0.,
	zoom_range=0.,
	channel_shift_range=0.,
	fill_mode='reflect',
	cval=0.,
	horizontal_flip=True,
	vertical_flip=True,
	rescale=None,
	preprocessing_function=None)
	return(generator)

class FFNN(object):
	def __init__(self, input_size, output_size):
		# Input
		input_layer = Input(shape=(input_size,))

		# Dense layers
		dense = Dense(100, activation='relu')(input_layer)
		dense = Dropout(0.25)(dense)
		dense = Dense(100, activation='relu')(dense)
		dense = Dropout(0.25)(dense)
		dense = Dense(100, activation='relu')(dense)

		output_layer = Dense(output_size, activation='softmax')(dense)

		self.output = output_layer
		self.input = input_layer

	def __str__(self):
		return """Simple feed-forward neural network with two layers."""

class SimpleCNN(object):
    def __init__(self, size, output_size, output='multilabel'):
        # Input
        image_input = Input(shape=(int(size),int(size), 3))

        # Convolutional layer
        conv_layer = Conv2D(2, (3,3), strides=(2,2), padding='valid', activation='relu')(image_input)
        conv_layer = MaxPooling2D()(conv_layer)
        conv_layer = Conv2D(32, (3,3), padding='valid', activation='relu')(conv_layer)
        conv_layer = MaxPooling2D()(conv_layer)

        # Flatten the output of the convolutional layer
        conv_output = Flatten()(conv_layer)

        # Stack Dense layer on top
        dense_layer = Dense(100, activation='relu')(conv_output)

        if output == 'multilabel':
            dense_output = Dense(output_size, activation='sigmoid')(dense_layer)

        elif output == 'multiclass':
            dense_output = Dense(output_size, activation='softmax')(dense_layer)

        self.input = image_input
        self.output = dense_output

    def __str__(self):
        return """Simple example CNN classifier."""

class UNet(object):
	def __init__(self, size, output_size):
		# Input
		image_input = Input(shape=(int(size),int(size), 3))

		# Preprocessing layer
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1, 1))(image_input))))
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1,1))(conv_0))))
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1,1))(conv_0))))

		def add_block(self, input, filtersize):
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (3,3), padding='same')(input)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (3,3), padding='same')(conv_)))
			output = MaxPooling2D(pool_size=(2, 2))(conv_)
			return(output)

		# Block 1
		conv_1 = add_block(self, conv_0, 32)
		conv_1_out = GlobalAveragePooling2D()(conv_1)

		# Block 1
		conv_2 = add_block(self, conv_1, 64)
		conv_2_out = GlobalAveragePooling2D()(conv_2)

		# Block 1
		conv_3 = add_block(self, conv_2, 128)
		conv_3_out = GlobalAveragePooling2D()(conv_3)

		# Concatenate all the outputs
		conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out])

		# Stack Dense layer on top
		dense = Activation('relu')(BatchNormalization()(Dense(512)(conv_concatenated)))
		#dense = Activation('relu')(BatchNormalization()(Dense(512)(conv_concatenated)))
		dense_output = Dense(output_size, activation='sigmoid')(dense)

		self.output = dense_output
		self.input = image_input

	def __str__(self):
		return """Downsampling part of a UNet model with a classifier stacked on top."""

class UNet_dstl(object):
	def __init__(self, size, output_size):
		# Input
		image_input = Input(shape=(int(size),int(size), 3))

		# Preprocessing layer
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1, 1))(image_input))))
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1,1))(conv_0))))
		conv_0 = Activation('relu')((BatchNormalization()(Conv2D(8, (1,1))(conv_0))))

		def add_block(self, input, filtersize):
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (3,3), padding='same')(input)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization()(Conv2D(filtersize, (3,3), padding='same')(conv_)))
			output = MaxPooling2D(pool_size=(2, 2))(conv_)
			return(output)

		# Block 1
		conv_1 = add_block(self, conv_0, 64)
		conv_1_out = GlobalMaxPooling2D()(conv_1)

		# Block 1
		conv_2 = add_block(self, conv_1, 64)
		conv_2_out = GlobalMaxPooling2D()(conv_2)

		# Block 1
		conv_3 = add_block(self, conv_2, 64)
		conv_3_out = GlobalMaxPooling2D()(conv_3)

		# Concatenate all the outputs
		conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out])

		# Stack Dense layer on top
		#dense = Activation('relu')(BatchNormalization()(Dense(512)(conv_concatenated)))
		#dense = Activation('relu')(BatchNormalization()(Dense(100)(conv_concatenated)))
		dense_output = Dense(output_size, activation='sigmoid')(conv_concatenated)

		self.output = dense_output
		self.input = image_input

	def __str__(self):
		return """Downsampling part of a UNet model with a classifier stacked on top.
		Some adaptations according to https://deepsense.io/deep-learning-for-satellite-imagery-via-image-segmentation/:
			* same number of channels across scales
			* model is fully convolutional
			* max pooling and not avg poolin"""

class SimpleNet64_2_plus_par(object):
	def __init__(self, size, output_size):
		# Input
		image_input = Input(shape=(int(size),int(size), 4))

		# Preprocessing layer
		def preprocessing(self, input, filtersize):
			conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(image_input))))
			conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_0))))
			conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_0))))

			return(conv_0)

		def add_block(self, input, filtersize):
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(input)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(conv_)))
			output = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv_)
			return(output)

		## RGB
		conv_0_rgb = preprocessing(self, image_input[:,:,:,:3], 8)

		# Block 1
		conv_1_rgb = add_block(self, conv_0_rgb, 32)
		conv_1_rgb_out = GlobalAveragePooling2D()(conv_1_rgb)

		# Block 2
		conv_2_rgb = add_block(self, conv_1_rgb, 32)
		conv_2_rgb_out = GlobalAveragePooling2D()(conv_2_rgb)

		# Block 3
		conv_3_rgb = add_block(self, conv_2_rgb, 64)
		conv_3_rgb = Dropout(0.25)(conv_3_rgb)
		conv_3_rgb_out = GlobalAveragePooling2D()(conv_3_rgb)

		# Block 4
		conv_4_rgb = add_block(self, conv_3_rgb, 128)
		conv_4_rgb = Dropout(0.5)(conv_4_rgb)
		conv_4_rgb_out = GlobalAveragePooling2D()(conv_4_rgb)

		## NIR

		conv_0_nir = preprocessing(self, image_input[:,:,:,3], 8)
		# Block 1
		conv_1_nir = add_block(self, conv_0_nir, 32)
		conv_1_nir_out = GlobalAveragePooling2D()(conv_1_nir)

		# Block 2
		conv_2_nir = add_block(self, conv_1_nir, 32)
		conv_2_nir_out = GlobalAveragePooling2D()(conv_2_nir)

		# Block 3
		conv_3_nir = add_block(self, conv_2_nir, 64)
		conv_3_nir = Dropout(0.25)(conv_3_nir)
		conv_3_nir_out = GlobalAveragePooling2D()(conv_3_nir)

		# Block 4
		conv_4_nir = add_block(self, conv_3_nir, 128)
		conv_4_nir = Dropout(0.5)(conv_4_nir)
		conv_4_nir_out = GlobalAveragePooling2D()(conv_4_nir)

		# Concatenate all the outputs
		conv_concatenated = k.layers.concatenate([conv_1_rgb_out, conv_2_rgb_out, conv_3_rgb_out, conv_4_rgb_out, conv_1_nir_out, conv_2_nir_out, conv_3_nir_out, conv_4_nir_out])

		# Stack Dense layer on top
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(conv_concatenated)))
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(dense)))
		dense_output = Dense(output_size, activation='sigmoid')(dense)

		self.output = dense_output
		self.input = image_input


class SimpleNet64_2_plus(object):
	def __init__(self, size, output_size):
		# Input
		image_input = Input(shape=(int(size),int(size), 4))

		# Preprocessing layer
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(image_input))))
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))

		def add_block(self, input, filtersize):
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(input)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(conv_)))
			output = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv_)
			return(output)

		# Block 1
		conv_1 = add_block(self, conv_0, 32)
		conv_1_out = GlobalAveragePooling2D()(conv_1)

		# Block 2
		conv_2 = add_block(self, conv_1, 32)
		conv_2_out = GlobalAveragePooling2D()(conv_2)

		# Block 3
		conv_3 = add_block(self, conv_2, 64)
		conv_3 = Dropout(0.25)(conv_3)
		conv_3_out = GlobalAveragePooling2D()(conv_3)

		# Block 4
		conv_4 = add_block(self, conv_3, 128)
		conv_4 = Dropout(0.5)(conv_4)
		conv_4_out = GlobalAveragePooling2D()(conv_4)

		# Concatenate all the outputs
		conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out, conv_4_out])

		# Stack Dense layer on top
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(conv_concatenated)))
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(dense)))
		dense_output = Dense(output_size, activation='sigmoid')(dense)

		self.output = dense_output
		self.input = image_input


class SimpleNet64_2(object):
    def __init__(self, size, output_size, output='multilabel'):
        # Input
        image_input = Input(shape=(int(size),int(size), 3))

        # Preprocessing layer
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(image_input))))
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))

        def add_block(self, input, filtersize):
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(input)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(conv_)))
            output = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv_)
            return(output)

        # Block 1
        conv_1 = add_block(self, conv_0, 32)
        conv_1_out = GlobalAveragePooling2D()(conv_1)

        # Block 2
        conv_2 = add_block(self, conv_1, 32)
        conv_2_out = GlobalAveragePooling2D()(conv_2)

        # Block 3
        conv_3 = add_block(self, conv_2, 64)
        conv_3 = Dropout(0.25)(conv_3)
        conv_3_out = GlobalAveragePooling2D()(conv_3)

        # Block 4
        conv_4 = add_block(self, conv_3, 128)
        conv_4 = Dropout(0.5)(conv_4)
        conv_4_out = GlobalAveragePooling2D()(conv_4)

        # Concatenate all the outputs
        conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out, conv_4_out])

        # Stack Dense layer on top
        dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(conv_concatenated)))
        dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(dense)))
        if output == 'multilabel':
            dense_output = Dense(output_size, activation='sigmoid')(dense)
        if output == 'multiclass':
            dense_output = Dense(output_size, activation='softmax')(dense)

        self.output = dense_output
        self.input = image_input


class SimpleNet_multitask(object):
	def __init__(self, size, output_size_multiclass, output_size_multilabel):
		# Input
		image_input = Input(shape=(int(size),int(size), 3))

		# Preprocessing layer
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1, 1))(image_input))))
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))
		conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))

		def add_block(self, input, filtersize):
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(input)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
			conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(conv_)))
			output = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv_)
			return(output)

		# Block 1
		conv_1 = add_block(self, conv_0, 32)
		conv_1_out = GlobalAveragePooling2D()(conv_1)

		# Block 2
		conv_2 = add_block(self, conv_1, 32)
		conv_2_out = GlobalAveragePooling2D()(conv_2)

		# Block 3
		conv_3 = add_block(self, conv_2, 64)
		conv_3 = Dropout(0.25)(conv_3)
		conv_3_out = GlobalAveragePooling2D()(conv_3)

		# Block 4
		conv_4 = add_block(self, conv_3, 128)
		conv_4 = Dropout(0.5)(conv_4)
		conv_4_out = GlobalAveragePooling2D()(conv_4)

		# Concatenate all the outputs
		conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out, conv_4_out])

		# Stack Dense layer on top
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(conv_concatenated)))
		dense = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Dense(512)(dense)))
		dense_output_multiclass = Dense(output_size_multiclass, activation='softmax')(dense)
		dense_output_multilabel  = Dense(output_size_multilabel, activation='sigmoid')(dense)

		self.output_multiclass = dense_output_multiclass
		self.output_multilabel = dense_output_multilabel
		self.input = image_input

class SimpleCNN_joint_GFM(object):
    ##NOTE: THIS IS STILL UGLY CODE, JUST TO TRY THE CONCEPT
    def __init__(self, size, field_sizes):

        # Input
        image_input = Input(shape=(int(size),int(size), 3))

        # Convolutional layer
        conv_layer = Conv2D(2, (3,3), strides=(2,2), padding='valid', activation='relu')(image_input)
        conv_layer = MaxPooling2D()(conv_layer)
        conv_layer = Conv2D(32, (3,3), padding='valid', activation='relu')(conv_layer)
        conv_layer = MaxPooling2D()(conv_layer)

        # Flatten the output of the convolutional layer
        conv_output = Flatten()(conv_layer)

        # Stack 17 multiclass classification problems on top

        multiclass_0 = Dense(field_sizes[0], activation='softmax')(conv_output)
        multiclass_1 = Dense(field_sizes[1], activation='softmax')(conv_output)
        multiclass_2 = Dense(field_sizes[2], activation='softmax')(conv_output)
        multiclass_3 = Dense(field_sizes[3], activation='softmax')(conv_output)
        multiclass_4 = Dense(field_sizes[4], activation='softmax')(conv_output)
        multiclass_5 = Dense(field_sizes[5], activation='softmax')(conv_output)
        multiclass_6 = Dense(field_sizes[6], activation='softmax')(conv_output)
        multiclass_7 = Dense(field_sizes[7], activation='softmax')(conv_output)
        multiclass_8 = Dense(field_sizes[8], activation='softmax')(conv_output)
        multiclass_9 = Dense(field_sizes[9], activation='softmax')(conv_output)
        multiclass_10 = Dense(field_sizes[10], activation='softmax')(conv_output)
        multiclass_11 = Dense(field_sizes[11], activation='softmax')(conv_output)
        multiclass_12 = Dense(field_sizes[12], activation='softmax')(conv_output)
        multiclass_13 = Dense(field_sizes[13], activation='softmax')(conv_output)
        multiclass_14 = Dense(field_sizes[14], activation='softmax')(conv_output)
        multiclass_15 = Dense(field_sizes[15], activation='softmax')(conv_output)
        multiclass_16 = Dense(field_sizes[16], activation='softmax')(conv_output)

        dense_output = [multiclass_0,
                        multiclass_1,
                        multiclass_2,
                        multiclass_3,
                        multiclass_4,
                        multiclass_5,
                        multiclass_6,
                        multiclass_7,
                        multiclass_8,
                        multiclass_9,
                        multiclass_10,
                        multiclass_11,
                        multiclass_12,
                        multiclass_13,
                        multiclass_14,
                        multiclass_15,
                        multiclass_16]

        self.input = image_input
        self.output = dense_output

class GFM_top_classifier(object):
    """Input should be flat."""

    def __init__(self, shape, field_sizes, nodes):
        # Input
        network_input = Input(shape[1:])

        max_pooled_input = GlobalMaxPooling2D()(network_input)
        # Some shared layers
        shared_dense_1 = Dense(nodes, activation='relu')(max_pooled_input)
        shared_dense_1 = Dropout(0.10)(shared_dense_1)
        shared_dense_2 = Dense(nodes, activation='relu')(shared_dense_1)
        shared_dense_2 = Dropout(0.10)(shared_dense_2)
        shared_dense_3 = Dense(nodes, activation='relu')(shared_dense_2)
        shared_dense_3 = Dropout(0.10)(shared_dense_3)

        # Stack 17 multiclass classification problems on top
        multiclass_0 = Dense(field_sizes[0], activation='softmax')(shared_dense_3)
        multiclass_1 = Dense(field_sizes[1], activation='softmax')(shared_dense_3)
        multiclass_2 = Dense(field_sizes[2], activation='softmax')(shared_dense_3)
        multiclass_3 = Dense(field_sizes[3], activation='softmax')(shared_dense_3)
        multiclass_4 = Dense(field_sizes[4], activation='softmax')(shared_dense_3)
        multiclass_5 = Dense(field_sizes[5], activation='softmax')(shared_dense_3)
        multiclass_6 = Dense(field_sizes[6], activation='softmax')(shared_dense_3)
        multiclass_7 = Dense(field_sizes[7], activation='softmax')(shared_dense_3)
        multiclass_8 = Dense(field_sizes[8], activation='softmax')(shared_dense_3)
        multiclass_9 = Dense(field_sizes[9], activation='softmax')(shared_dense_3)
        multiclass_10 = Dense(field_sizes[10], activation='softmax')(shared_dense_3)
        multiclass_11 = Dense(field_sizes[11], activation='softmax')(shared_dense_3)
        multiclass_12 = Dense(field_sizes[12], activation='softmax')(shared_dense_3)
        multiclass_13 = Dense(field_sizes[13], activation='softmax')(shared_dense_3)
        multiclass_14 = Dense(field_sizes[14], activation='softmax')(shared_dense_3)
        multiclass_15 = Dense(field_sizes[15], activation='softmax')(shared_dense_3)
        multiclass_16 = Dense(field_sizes[16], activation='softmax')(shared_dense_3)

        dense_output = [multiclass_0,
                        multiclass_1,
                        multiclass_2,
                        multiclass_3,
                        multiclass_4,
                        multiclass_5,
                        multiclass_6,
                        multiclass_7,
                        multiclass_8,
                        multiclass_9,
                        multiclass_10,
                        multiclass_11,
                        multiclass_12,
                        multiclass_13,
                        multiclass_14,
                        multiclass_15,
                        multiclass_16]

        self.input = network_input
        self.output = dense_output

class SimpleNet64_joint_GFM(object):

    def __init__(self, size, field_sizes):

        # Input
        image_input = Input(shape=(int(size),int(size), 3))

        # Preprocessing layer
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(image_input))))
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))
        conv_0 = Activation('relu')((BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(8, (1,1))(conv_0))))

        def add_block(self, input, filtersize):
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(input)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (1,1))(conv_)))
            conv_ = Activation('relu')(BatchNormalization(epsilon=0.0001, momentum=0.1)(Conv2D(filtersize, (3,3), padding='same')(conv_)))
            output = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv_)
            return(output)

        # Block 1
        conv_1 = add_block(self, conv_0, 32)
        conv_1_out = GlobalAveragePooling2D()(conv_1)

        # Block 2
        conv_2 = add_block(self, conv_1, 32)
        conv_2_out = GlobalAveragePooling2D()(conv_2)

        # Block 3
        conv_3 = add_block(self, conv_2, 64)
        conv_3 = Dropout(0.25)(conv_3)
        conv_3_out = GlobalAveragePooling2D()(conv_3)

        # Block 4
        conv_4 = add_block(self, conv_3, 128)
        conv_4 = Dropout(0.5)(conv_4)
        conv_4_out = GlobalAveragePooling2D()(conv_4)

        # Concatenate all the outputs
        conv_concatenated = k.layers.concatenate([conv_1_out, conv_2_out, conv_3_out, conv_4_out])
        dense_shared = Dense(512, activation='relu')(conv_concatenated)
        dense_shared = Dense(512, activation='relu')(dense_shared)
        # Stack 17 multiclass classification problems on top

        multiclass_0 = Dense(field_sizes[0], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_1 = Dense(field_sizes[1], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_2 = Dense(field_sizes[2], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_3 = Dense(field_sizes[3], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_4 = Dense(field_sizes[4], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_5 = Dense(field_sizes[5], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_6 = Dense(field_sizes[6], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_7 = Dense(field_sizes[7], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_8 = Dense(field_sizes[8], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_9 = Dense(field_sizes[9], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_10 = Dense(field_sizes[10], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_11 = Dense(field_sizes[11], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_12 = Dense(field_sizes[12], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_13 = Dense(field_sizes[13], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_14 = Dense(field_sizes[14], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_15 = Dense(field_sizes[15], activation='softmax')(Dense(100, activation='relu')(dense_shared))
        multiclass_16 = Dense(field_sizes[16], activation='softmax')(Dense(100, activation='relu')(dense_shared))

        dense_output = [multiclass_0,
                        multiclass_1,
                        multiclass_2,
                        multiclass_3,
                        multiclass_4,
                        multiclass_5,
                        multiclass_6,
                        multiclass_7,
                        multiclass_8,
                        multiclass_9,
                        multiclass_10,
                        multiclass_11,
                        multiclass_12,
                        multiclass_13,
                        multiclass_14,
                        multiclass_15,
                        multiclass_16]


        self.output = dense_output
        self.input = image_input
