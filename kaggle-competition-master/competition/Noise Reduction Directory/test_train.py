import numpy as np
import sys

images = np.load('../Dataset/train_images.npy')
imageIndex = int(sys.argv[1])
entries = len(images[imageIndex][1])
for i in range(entries): sys.stdout.write(str(int(images[imageIndex][1][i]) & 0xFF) + ' ')
sys.stdout.flush()
