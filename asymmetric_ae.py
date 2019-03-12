from template_ae import ModelTemplate

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense

class AEModel(ModelTemplate):
    def __init__(self):
        self.sample_rate = 22050
        self.input_size = round(self.sample_rate*0.1)

        self.epochs = 100
        self.batch_size = 50

        learning_rate = 0.00001
        self.optimizer = Adam(lr=learning_rate)

        self.loss_function = 'mse'
        
        compression_rate = 0.1
        comp_size = round(compression_rate * self.input_size)

        input_layer = Input(shape=(self.input_size,))
        
        encoded = Dense(comp_size*5-120, activation='sigmoid')(input_layer)
        encoded = Dense(comp_size*2, activation='sigmoid')(encoded)
        encoded = Dense(comp_size, activation='sigmoid')(encoded)  
        decoded = Dense(self.input_size, activation='sigmoid')(encoded)
        
        decoded = Dense(self.input_size, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        encoded_input = Input(shape=(comp_size,))
        decoder_layer = self.autoencoder.layers[-1]
        
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

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
    