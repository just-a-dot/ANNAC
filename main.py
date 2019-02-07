import preprocessor
import postprocessor
import sys

wavs = []
for arg in sys.argv[1:]:
    wavs.append((arg, preprocessor.wavToNumpy(arg)))
for nm, data in wavs:
    postprocessor.numpyToWav(data, 'out-' + nm)
