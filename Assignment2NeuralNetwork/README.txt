ReadMe File

The given folder contains three files and one folder
1. winequality-red.csv
2. NeuralNet.py
3. requirements.txt
4. readings.csv

1. winequality-red.csv
-------------------
This is the csv file containing the dataset used for the assignment. Please keep this dataset in the same folder as the code file (NeuralNet.py)

2. NeuralNet.py
-----------------------
This is the actual code file and accepts command line arguments to get the inputs as follows :
--i. To run the file for sigmoid, tanh and relu use following commands

	python NeuralNet.py sigmoid True winequality-red.csv
  python NeuralNet.py tanh True winequality-red.csv
  python NeuralNet.py relu True winequality-red.csv

Here first argument denotes the activation function, the second argument denotes
whether the program will print graph of MSE vs Iteration.
If value is True it will print the graph, else it will not print the graph. Default value is False.
Third argument denotes the datasetfile to be used.
This code reads the dataset file, preprocesses it and splits it into two parts in 80:20 split


3. requirements.txt
--------------------
This code uses only numpy, pandas and matplotlib
This file contains the list of libraries required for the code to run.

4. readings.csv
-----------------
This folder contains csv file that contains the readings for all three activation
functions with epoch 1 to 1000 and learning rate from 0.001 to 0.010
