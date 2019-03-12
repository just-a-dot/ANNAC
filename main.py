import preprocessor
import postprocessor
import sys
import os

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Flatten, TimeDistributed, MaxPooling1D, Reshape, BatchNormalization, UpSampling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from random import shuffle
from math import ceil

from multiprocessing import Pool

if len(sys.argv) < 2:
    print("Usage: main.py followed by a list of soundfiles")
    exit()

# Size of the samples. Decides network size, ram usage increses exponentially with this
#size = round(22050 * 0.5)
size = 1 
#chunk_size = round(22050*0.1)
chunk_size = 2000
compression_rate = 0.1
compressed_size = round(compression_rate * chunk_size)
song_length = 22050*29


#Read the files given as arguments
wavs = []
names = []
i = 1
for arg in sys.argv[1:]:
    names.append(arg)
def preprocess_par(song):
    return preprocessor.wavToNumpy(song, size, song_length)
with Pool() as p:
    res_temp = p.map(preprocess_par, sys.argv[1:])
    wavs = res_temp
    print(len(wavs))
    #wavs = sum(res_temp, [])


print(chunk_size)

if os.path.isfile('x_train.npy') and os.path.isfile('x_test.npy'):
    print('''Found saved traning data!
          Using given arguments for processing, not training.''')
    with open('x_train.npy', 'rb') as f:
        x_train = np.load(f)
        print(x_train.shape)
    with open('x_test.npy', 'rb') as f:
        x_test = np.load(f)
        print(x_test.shape)
else:
    sequence = np.zeros((ceil(song_length/chunk_size)*len(wavs), chunk_size, size))
    print(sequence.shape)
    print(len(wavs[0]))
    song_i = 0
    for song in wavs:
        sample_i = 0
        chunk_i = 0
        for sample in song:
            if(len(sample) == size):
                sequence[(song_i*(song_length//chunk_size)) + chunk_i][sample_i] = sample
                sample_i += 1
            if sample_i == chunk_size:
                chunk_i += 1
                sample_i = 0
        song_i += 1
    
    shuffle(sequence)
    print(sequence.shape)
    x_train = np.asarray(sequence[:(round(0.9*len(sequence)))])
    x_test = np.asarray(sequence[(round(0.9*len(sequence))):])
    print('Saving training data')
    with open('x_train.npy', 'wb') as f:
        np.save(f, x_train)
    with open('x_test.npy', 'wb') as f:
        np.save(f, x_test)


inputs = Input(shape=(chunk_size, size))
encoded = Conv1D(32, 3, activation='relu')(inputs)
encoded = MaxPooling1D(2)(encoded)
encoded = Conv1D(16, 3, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = MaxPooling1D(2)(encoded)
encoded = BatchNormalization()(encoded)
encoded = Flatten()(encoded)
encoded = Dense(compressed_size, activation='relu')(encoded)

decoded = Reshape((compressed_size, size))(encoded)
decoded = Conv1D(16,3, activation='relu')(decoded)
decoded = UpSampling1D(2)(decoded)
decoded = Conv1D(32,11, activation='relu')(decoded)
decoded = UpSampling1D(2)(decoded)
decoded = Flatten()(decoded)
decoded = Dense(chunk_size, activation='sigmoid')(decoded)
print(decoded.shape)
decoded = Reshape((chunk_size, size))(decoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mse')


# Save the weights after each epoch if they improved
filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min')
stopping = EarlyStopping(patience=10)
callback_list = [checkpoint, stopping]


autoencoder.fit(
    x_train, 
    x_train, 
    epochs=3000, 
    batch_size = 100, 
    validation_data=(x_test, x_test),
    callbacks = callback_list)
autoencoder.save('model.h5')

#Reassemble wavs
for i in range(0, len(wavs)):
    print("Processing song " + str(i) + " of " + str(len(wavs)))
    print(wavs[i].shape)
    encoded_wav = []
    for chunk_i in range(0, song_length):
        start_index = ((song_length//chunk_size)*i)+(chunk_i*chunk_size)
        temp = wavs[i][start_index:start_index + chunk_size]
        if len(temp) != chunk_size:
            continue
        temp = temp.reshape(1, chunk_size, 1)
        encoded_wav.append(autoencoder.predict((temp), batch_size=1))
        #decoded_wav = decoder.predict(encoded_wav)
    postprocessor.numpyToWav(np.array(encoded_wav), names[i] + '-out.wav')

