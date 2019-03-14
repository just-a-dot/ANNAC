from template_ae import ModelTemplate

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, TimeDistributed, MaxPooling1D, Reshape, BatchNormalization, UpSampling1D

class AEModel(ModelTemplate):
    def __init__(self):
        self.input_size = 1 
        self.optimizer = Adam(lr=0.0001)
        self.loss_function = 'mse'
        self.sample_rate = 22050

        self.epochs = 1000
        self.batch_size = 100

        #chunk_size = round(22050*0.1)
        self.chunk_size = 2000
        compression_rate = 0.1
        compressed_size = round(compression_rate * self.chunk_size)

        inputs = Input(shape=(self.chunk_size, self.input_size))
        encoded = Conv1D(32, 3, activation='relu')(inputs)
        encoded = MaxPooling1D(2)(encoded)
        encoded = Conv1D(16, 3, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = MaxPooling1D(2)(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(compressed_size, activation='relu')(encoded)

        decoded = Reshape((compressed_size, self.input_size))(encoded)
        decoded = Conv1D(16,3, activation='relu')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(32,11, activation='relu')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Flatten()(decoded)
        decoded = Dense(self.chunk_size, activation='sigmoid')(decoded)

        decoded = Reshape((self.chunk_size, self.input_size))(decoded)

        self.autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)
        self.decoder = Model(inputs, encoded)

    def get_epochs(self):
        return self.epochs
        
    def get_input_size(self):
        return self.input_size

    def get_batch_size(self):
        return self.batch_size

    def get_optimizer(self):
        return self.optimizer

    def get_loss_function(self):
        return self.loss_function

    def get_autoencoder(self):
        return self.autoencoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_sample_rate(self):
        return self.sample_rate

    def use_lstm(self):
        return True
    