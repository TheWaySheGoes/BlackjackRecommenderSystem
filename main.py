#This file is the main entrance to the program.
#Here everything is comming together
#
import threading
#from show_screen import screen
import test as t
import gui
import genetic_algo



#thread for gui
gui=gui.GUI()
gui.start()
#thread for gen (just for testing multithreading)
gen=genetic_algo.Genetic_algo()
gen.start()

gen.learn()
gen.best_solution()





