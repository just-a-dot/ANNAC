import sys
import preprocessor

print(sys)
print(sys.argv)

if len(sys.argv) < 2 and not os.path.isfile('x-train.npy'):
    print("Usage: main.py followed by a list of soundfiles")
    print(sys.argv)
    exit()

size = round(22050 * 0.5)
    
wavs = []
names = []
i = 0
for arg in sys.argv[1:]:
    print("Reading file " + str(i+1) + "/" + str(len(sys.argv[1:])) + "              ", end='\r')
    wavs.append(preprocessor.wavToNumpy(arg, size))
    names.append(arg)
    i += 1

    print (wavs)