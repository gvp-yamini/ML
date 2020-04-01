ReadMe File

The given folder contains three files and one folder
1. datasetfile.csv
2. LinearRegression.py
3. requirements.txt
4. ReadingsFolder

1. datasetfile.csv
-------------------
This is the csv file containing the dataset used for the assignment. Please keep this dataset in the same folder as the code file (LinearRegression.py)

2. LinearRegression.py
-----------------------
This is the actual code file and accepts command line arguments to get the inputs as follows :
--i. To run the file for the first time use following command

	python LinearRegression.py False datasetfile.csv

Here first argument denotes whether the program will print graph of MSE vs Iteration. If value is True it will print the graph, else it will not print the graph. Default value is False.
Second argument denotes the datasetfile to be used.This code reads the dataset file, preprocesses it and splits it into two parts train.csv and test.csv


--ii. Running file afterwards , use the following command

	python LinearRegression.py

Here the code will assume that train.csv and test.csv already exist in the folder.

-iii. Run file with graphs

	python LinearRegression.py True

Here the code will assume that train.csv and test.csv already exist in the folder.
Here first argument denotes whether the program will print graph of MSE vs Iteration.
If value is True it will print the graph, else it will not print the graph.
Default value is False.

3. requirements.txt
--------------------
This file contains the list of libraries required for the code to run.

4. ReadingsFolder
-----------------
This folder contains 2 csv files, one for Part1 readings and other for Part2 readings.
