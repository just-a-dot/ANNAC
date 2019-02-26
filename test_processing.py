import preprocessor as pre
import postprocessor as post

sample_rate = 22050
size = round(sample_rate * 0.5) 
nparr = pre.wavToNumpy('dataset/blues/blues.00000.au', size)
post.numpyToWav(nparr, 'reconstructed.au')    # fails, but should work!