import preprocessor
import postprocessor
import sys

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D

if len(sys.argv) < 2:
    print("Usage: main.py followed by a list of soundfiles")
    exit()

# Size of the samples. Decides network size, ram usage increses exponentially with this
#size = round(22050 * 0.5)
size = 1


#Read the files given as arguments
wavs = []
names = []
i = 1
for arg in sys.argv[1:]:
    print("Reading file " + str(i) + "/" + str(len(sys.argv[1:])))
    wavs.append(preprocessor.wavToNumpy(arg, size))
    names.append(arg)
    i += 1


song_length = (22050*4)

sequence = np.zeros((len(wavs), song_length, size))
song_i = 0
for song in wavs:
    sample_i = 0
    for sample in song:
        if(len(sample) == size):
            sequence[song_i][sample_i] = sample
        sample_i += 1
        if sample_i == song_length:
            break
    song_i += 1

inputs = Input(shape=(song_length,size)) 

# encoder
x = Conv1D(song_length//10, 10, activation='relu', padding='same')(inputs)
x = MaxPooling1D()(x)
x = Conv1D(song_length//10, 10, activation='relu', padding='same')(x)
x = MaxPooling1D()(x)
x = Conv1D(song_length//10, 10, activation='relu', padding='same')(x)
encoded = MaxPooling1D()(x)

# decoder
x = Conv1D(song_length//10, 10, activation='relu', padding='same')(encoded)
x = UpSampling1D()(x)
x = Conv1D(song_length//10, 10, activation='relu', padding='same')(x)
x = UpSampling1D()(x)
x = Conv1D(song_length//10, 10, activation='relu')(x)
x = UpSampling1D()(x)
decoded = Conv1D(1, 10, activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

x_train = np.asarray(sequence[:(round(0.9*len(sequence)))])
x_test = np.asarray(sequence[(round(0.9*len(sequence))):])

autoencoder.fit(x_train, x_train, epochs=1, batch_size = 1, validation_data=(x_test, x_test))

#Reassemble wavs
for i in range(0, len(wavs)):
    print("Processing song " + str(i) + " of " + str(len(wavs)))
    print(wavs[i].shape)
    temp = wavs[i][:song_length].reshape(1, song_length, 1)
    encoded_wav = autoencoder.predict((temp), batch_size=1)
    #decoded_wav = decoder.predict(encoded_wav)
    print(encoded_wav.shape)
    postprocessor.numpyToWav(encoded_wav, names[i] + '-out.wav')