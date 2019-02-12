import preprocessor
import postprocessor
import sys

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense

if len(sys.argv) < 2:
    print("Usage: main.py followed by a list of soundfiles")
    exit()

# Size of the samples. Decides network size, ram usage increses exponentially with this
size = round(22050 * 1)

input_layer = Input(shape=(size,))

encoded = Dense(size//10, activation='relu', kernel_initializer='random_uniform')(input_layer)

decoded = Dense(size, activation='sigmoid', kernel_initializer='random_uniform')(encoded)

autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

encoded_input = Input(shape=(size//10,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#Read the files given as arguments
wavs = []
names = []
i = 0
for arg in sys.argv[1:]:
    print("Reading file " + str(i) + "/" + str(len(sys.argv[1:])))
    wavs.append(preprocessor.wavToNumpy(arg, size))
    names.append(arg)
    i += 1

#Training happens here
wavs_trans = []
for song in wavs:
    for sample in song:
        if len(sample) != size:
            continue
        wavs_trans.append(sample)

x_train = np.asarray(wavs_trans[:(round(0.9*len(wavs_trans)))])
x_test = np.asarray(wavs_trans[(round(0.9*len(wavs_trans))):])


autoencoder.fit(x_train, x_train, epochs=1, validation_data=(x_test, x_test))

#Reassemble wavs
for i in range(0, len(wavs)):
    base = []
    for sample in wavs[i]:
        if len(sample) != size:
            continue
        encoded_wav = encoder.predict(np.expand_dims(sample, axis=0), batch_size=1)
        decoded_wav = decoder.predict(encoded_wav)
        base.append(decoded_wav)
    postprocessor.numpyToWav(base, names[i] + '-out.wav')
