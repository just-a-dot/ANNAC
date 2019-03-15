from template_ae import ModelTemplate

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed

class AEModel(ModelTemplate):
    def __init__(self):
        self.sample_rate = 22050
        self.input_size = 1
        self.chunk_size = 500 # round(self.sample_rate*0.1)

        self.epochs = 100
        self.batch_size = 50

        learning_rate = 0.00001
        self.optimizer = Adam(lr=learning_rate)

        self.loss_function = 'mse'
        
        compression_rate = 0.1

        comp_size = round(compression_rate * self.chunk_size)

        input_layer = Input(shape=(self.chunk_size,self.input_size))
        
        encoded = LSTM(comp_size)(input_layer)
        
        decoded = RepeatVector(self.chunk_size)(encoded)
        decoded = LSTM(1, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.input_size, activation='sigmoid'))(decoded)

        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # TODO: create a separate decoder
        encoded_input = Input(shape=(comp_size,))
        decoder_output = self.autoencoder.layers[-3](encoded_input)
        decoder_output = self.autoencoder.layers[-2](decoder_output)
        decoder_output = self.autoencoder.layers[-1](decoder_output)
        
        self.decoder = Model(encoded_input, decoder_output)

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
    