# Alvin Noor Hidayah-21091397016-B-UTS


import numpy as np

#inisialisasi varibel didalam input
inputs = [10, 12, 5, 9, 7, 1, 2, 13, 4, 16]
weights = [0.1, 0.9, -0.1, 0.2, -0.4, 0.5, 0.8, 0.91, 0.14, -0.17]
weights = [0.2, 0.6, -0.8, 0.3, -0.4, 0.24 ,0.12, 0.43, 0.81, -0.57]
weights = [0.3, 0.7, -0.7, 0.4, -0.3, 0.5, 0.8, 0.41, 0.88, -0.5]
weights = [0.4, 0.7, -0.1, 0.6, -0.2, 0.51, 0.23, 0.42, 0.77, -0.55]
weights = [0.5, 0.3, -0.2, 0.9, -0.1, 0.22, 0.11, 0.41, 0.66, -0.11]

#memasukkan bobot per neuron karena multi banyak
bias = [1, 2, 3, 4, 0.5]

#perhitungan ouput
outputs = np.dot(weights, inputs) + bias
print(outputs)

