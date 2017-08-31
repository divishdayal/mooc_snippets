import os
import numpy as np

GLOVE_DIR = '../animesh/glove/'

glove_vec = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_vec[word] = coefs
f.close()

