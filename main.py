import preprocessor
import postprocessor
import sys

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Flatten, TimeDistributed

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
print(song_length)

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

print(sequence.shape)

inputs = Input(shape=(song_length, size))
encoded = Conv1D(song_length//10, 10, input_shape=(song_length, size), padding='causal', activation='relu')(inputs)
#encoded = Flatten()(encoded)

decoded = TimeDistributed(Dense(1, activation='sigmoid'))(encoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer='RMSprop', loss='mse')


x_train = np.asarray(sequence[:(round(0.9*len(sequence)))])
x_test = np.asarray(sequence[(round(0.9*len(sequence))):])

autoencoder.fit(x_train, x_train, epochs=10, batch_size = 1, validation_data=(x_test, x_test))
autoencoder.save('model.h5')

#Reassemble wavs
for i in range(0, len(wavs)):
    print("Processing song " + str(i) + " of " + str(len(wavs)))
    print(wavs[i].shape)
    temp = wavs[i][:song_length].reshape(1, song_length, 1)
    encoded_wav = autoencoder.predict((temp), batch_size=1)
    #decoded_wav = decoder.predict(encoded_wav)
    print(encoded_wav.shape)
    postprocessor.numpyToWav(encoded_wav, names[i] + '-out.wav')

