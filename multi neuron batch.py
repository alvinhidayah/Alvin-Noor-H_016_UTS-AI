#ALVIN NOOR HIDAYAH-21091397016-B-UTS

import numpy as np

#memasukkan/insisialiasasi varibael input sebanyak 10 dengan batch 6
inputs = [[2.0, 1.0, 4.0, 2.2, 1.0, 0.6],
		 [1.0, 3.0,-1.0, 1.0, 2.0, 0.4],
		 [-1.2, 2.2, 3.4,-0.7, 4.2, 5.2],
         [1.2, 0.8, 0.3, 0.4, 0.6, 4.5],
         [3.1, 3.2, 3.3, 4.5, 6.1, 6.6],
         [1.2, 3.2, 3.4,-1.7, 4.2, 3.2],
         [-1.9, 2.3, 3.4,-0.7, 1.2, 3.2],
         [1.3, 1.2, 3.9,-0.7, 1.2, 5.8],
         [-3.2, 2.3, 3.2,-1.7, 9.2, 7.2],
         [1.4, 4.2, 3.4,-0.8, 4.2, 8.2],
         [-0.9, 2.7, 3.1,-0.6, 5.2, 7.2]]

 #banyak bobot dengan 5 neuron dan bacth 6      
weights =[[1.3, 1.2, 3.4,-0.7, 9.2, 0.4 ],
          [1.2, 2.9, 7.4,0.5, 9.2, 1.2],
          [8.2, 8.2, 3.4,-1.7, 8.2, 9.3],
          [7.2, 5.7, 3.4,0.4, 2.2, 9.9],
          [1.5, 2.9, 3.4,-0.1, 7.2, 9.3]]

#banyak di tiap2 bias, bias tergantung banyak weight juga
bias = [0.2, 0.4, 0.5, 0.7, 0.1]

#perhitungan ouput  menggunakan np dan di transpose
layer_outputs = np.dot(inputs, np.array(weights).T) + bias
#perkalian matrix 10x6 . 6x5 = 5x10
# jadi 10x6 itu diitung jml input + banyak batch
#dan 6x5 itu jml batch + banyak weights

print(layer_outputs)