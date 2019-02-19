import preprocessor
import postprocessor
import sys

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

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


song_length = (22050*5)
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
memory_size = 100

inputs = Input(shape=(song_length, size))
encoded = LSTM(memory_size)(inputs)

decoded = RepeatVector(song_length)(encoded)
decoded = LSTM(memory_size, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(1, activation='sigmoid'))(decoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer='RMSprop', loss='mse')

print(get_model_memory_usage(1, autoencoder))

x_train = np.asarray(sequence[:(round(0.9*len(sequence)))])
x_test = np.asarray(sequence[(round(0.9*len(sequence))):])

autoencoder.fit(x_train, x_train, epochs=100, batch_size = 1, validation_data=(x_test, x_test))
autoencoder.save('model.h5')

#Reassemble wavs
for i in range(0, len(wavs)):
    print("Processing song " + str(i) + " of " + str(len(wavs)))
    encoded_wav = autoencoder.predict(wavs[i].reshape((wavs[i].shape[0], )), batch_size=1)
    #decoded_wav = decoder.predict(encoded_wav)
    postprocessor.numpyToWav(encoded_wav, names[i] + '-out.wav')

