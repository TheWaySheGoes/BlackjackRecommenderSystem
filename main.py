#This file is the main entrance to the program.
#Here everything is comming together
#

import test as test
from Genetic_algo import Genetic_algo




test#first test



#testing the genetic module
data_vals=[[1.0, 1.0, 1.0],
    [2.0,2.0,2.0],
    [3.0,3.0,3.0],
    [4.0,4.0,4.0],
    [5.0,5.0,5.0]]
data_labels=[[0.0,0.0],
            [0.0,0.0],
            [2.0,2.0],
            [1.0,1.0],
            [1.0,1.0]]
output_size=len(data_labels[0])
print(output_size)
ga = Genetic_algo(500,output_size)

print(len(data_vals))
print(data_vals[0])


for i in range(0,len(data_vals)):
    ga.set_data(data_vals[i],data_labels[i])
    ga.learn()  
ga.predict([1.0,2.0,3.0])

###############












