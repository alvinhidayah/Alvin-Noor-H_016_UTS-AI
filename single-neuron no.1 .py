#ALVIN NOOR HIDAYAH-21091397016-B-UTS

import numpy as np

#memasukkan variabel/inisialisasi
inputs = [10, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, ]

#bobot neuron
weights = [0.1, 0.2, -0.1, 0.3, -0.4, 0.5, 0.8, 0.4, 0.9, -0.5]

#bias neuron
bias = 7.0

#perhitungan
outputs = np.dot(weights, inputs) + bias

print(outputs)


