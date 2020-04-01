#####################################################################################################################
#   CS 6375 - Assignment 1, Linear Regression using Gradient Descent
#   Assignment performed by
#   Ameya Kulkarni  (ANK190006)
#   Yamini Thota    (YRT190003)
#
#####################################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale as mms
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as err
import sys


class LinearRegression:
    def __init__(self,data_set_file_name=None):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        if data_set_file_name is not None:
            print("Reading file : ",data_set_file_name)
            self.initializeDataFromDataset(data_set_file_name)
        else:
            print("Dataset file input not given, program will consider existing train.csv and test.csv")

        # read data from train file
        df = self.getDataFrameFromCsv("train.csv")
        df.insert(0, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        print ("Number of Rows :",self.nrows,"\t Number of Features : ",self.ncols)
        self.X =  df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    def initializeDataFromDataset(self,data_set_file_name):
        """
        This function reads the csv file and creates a dataframe. It then preprocesses the data
        frame and splits it into 80:20 train test set
        :param data_set_file_name: Name of file containing entire dataset.
        :return: This function generates two files train.csv and test.csv
        """
        try:
            df = self.getDataFrameFromCsv(data_set_file_name)
            df = self.preProcessEntireDataset(df)
            self.splitDataSet(df)
        except Exception as excep:
            print("Exception occurred while reading the given dataset file : ", excep)
            exit(0)


    def splitDataSet(self,df):
        """
        This function will split the data frame in 80:20 ratio and 
        generate two files train.csv and test.csv
        :param df: the dataframe to split
        :return: This function generates two files train.csv and test.csv
        """
        df['index'] = df.reset_index().index
        testingDf = df.sample(frac=.20)
        df = df.drop(testingDf['index'])
        df = df.drop('index', 1)
        testingDf = testingDf.drop('index', 1)
        testingDf.to_csv('test.csv', index=False)
        df.to_csv('train.csv', index=False)
        print("Data set is preprocessed and split into train.csv and test.csv")


    def preProcessEntireDataset(self,df):
        """
        Method to preprocess the dataset. This method performs following preprocessing activities
        * Replaces the categorical values by numbers.
        * Replaces Date objects by Hour of the day
        * Removes the unnecessary column named 'weather_description'
        * Normalizes the values using MinMax normalization in specific columns 
        :param df: The data frame to preprocess
        :return: processed dataframe
        """
        df = self.replaceCategoricalValues(df)
        df = self.replaceDateTimeByHourOfDay(df)
        if 'weather_description' in df.columns:
            df = self.removeSpecifiedColumn(df, "weather_description")
        df = self.normalizeColumnsUsingMinMax(df, ['temp', 'rain_1h', 'clouds_all', 'traffic_volume'])
        return df


    def replaceCategoricalValues(self,df):
        """
        Method to replace the categorical values by numbers
        :param df: DataFrame to process
        :return: Processed dataframe
        """
        if 'weather_main' in df.columns:
            df['weather_main'] = df['weather_main'].map({
                'Clouds': 0,
                'Clear': 1,
                'Drizzle': 2,
                'Fog': 3,
                'Haze': 4,
                'Mist': 5,
                'Rain': 6,
                'Smoke': 7,
                'Snow': 8,
                'Squall': 9,
                'Thunderstorm': 10,
            }).fillna(df['weather_main'])

        if 'holiday' in df.columns:
            df['holiday'] = df['holiday'].map({
                'None': 0,
                'Columbus Day': 1,
                'Veterans Day': 2,
                'Thanksgiving Day': 3,
                'Christmas Day': 4,
                'New Years Day': 5,
                'Washingtons Birthday': 6,
                'Memorial Day': 7,
                'Independence Day': 8,
                'State Fair': 9,
                'Labor Day': 10,
                'Martin Luther King Jr Day': 11
            }).fillna(df['holiday'])
        return df

    def replaceDateTimeByHourOfDay(self,df):
        """
        Method to replace date time by hour of the day
        :param df: Dataframe to process
        :return: Processed dataframe
        """
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')
        df['date_time'] = df['date_time'].dt.hour
        return df

    def removeSpecifiedColumn(self,df,columnName):
        """
        Method to remove specific column from the dataframe
        :param df: Dataframe to proces
        :param columnName: Name of column to remove
        :return: Processed dataframe
        """
        return df.drop(columns=columnName)

    def normalizeColumnsUsingMinMax(self,df,columnNames):
        """
        Method to normalize the data in specific columns using minmax
        :param df: Dataframe to process
        :param columnNames: Names of columns to normalize
        :return: Processed dataframe
        """
        df[columnNames] = mms(df[columnNames])
        return df

    def getDataFrameFromCsv(self,csvFileName):
        """
        Method to get dataframe from csv file
        :param csvFileName: 
        :return: 
        """
        return pd.read_csv(csvFileName)

    def train(self, epochs, learning_rate,do_plot_graph=False):
        """
        Method to perform gradient descent using given epochs and learning rate
        :param epochs: Number of epochs
        :param learning_rate: Learning rate value
        :param do_plot_graph: Flag to determin whether to plot graph of mse vs iteration
        :return: Returnes updated weights, error, convergence iteration number and mean square error
        """
        # Perform Gradient Descent
        mseArray = np.empty([1])        # Array to keep track of training data mean square error after every iteration
        msetestArray = np.empty([1])    # Array to keep track of test data mean square error after every iteration
        iterationArray = np.empty([1])  # Array to keep corresponding iteration index
        error=0
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y
            self.W = self.W - (1 / float(self.nrows)) * learning_rate * np.dot(self.X.T, error)
            # Make prediction on training data and get back error
            localmse,localRSquare = model.predict("train.csv")
            # Make prediction on test data and get back error
            localmsetest, localRSquaretest = model.predict("test.csv")
            mseArray = np.append(mseArray,localmse.flatten(),0)
            msetestArray = np.append(msetestArray,localmsetest.flatten(),0)
            iterationArray = np.append(iterationArray,i)
            # when more that two iterations are done then apply convergence condition
            # converge when difference in mean square error is less than 0.001
            if mseArray.size>2 and mseArray[mseArray.size-2]-mseArray[mseArray.size-1] < 0.001 :
                print("Converging at epoch ",i)
                break
        iterationArray =  np.delete(iterationArray,0)
        mseArray = np.delete(mseArray, 0)
        msetestArray = np.delete(msetestArray, 0)
        # plot the graph of test and training error against iterations
        if do_plot_graph:
            plt.plot(iterationArray,mseArray,linestyle='-',marker='o')
            plt.plot(iterationArray,msetestArray,linestyle='-',marker='^')
            plt.xlabel("Iteration")
            plt.ylabel("Mean Square Error")
            plt.title("Mean Square Error(MSE) vs Iteration Convergence condition as difference in root mse < 0.001")
            plt.legend(['MSE on Training Data', 'MSE on Test Data'], loc='upper right')
            plt.show()
        localmse, localRSquare = self.predict("train.csv")
        return self.W, error,i,localmse.flatten()[0]

    def predict(self, datasetfile):
        """
        Method to run prediction using trained weights on examples in given datasetfile
        :param datasetfile: File name to get data for running prediction
        :return: Returns meas square error and R Square
        """
        dataFrame = pd.read_csv(datasetfile)
        dataFrame.insert(0, "X0", 1)
        nrows, ncols = dataFrame.shape[0], dataFrame.shape[1]
        testX = dataFrame.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = dataFrame.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        #print np.dot(error.T,error)
        mse = np.sqrt(((1/float(2*nrows)) * np.dot(np.transpose(error), error)))
        #mse = np.dot(error.T, error)
        RSS = np.dot(np.transpose(error), error)
        TSS = self.findTSS()
        rSquare = 1 - (RSS/TSS)
        return mse,rSquare

    def findTSS(self):
        """
        Method to calculate Total Sum of Squares
        :return: returns TSS value
        """
        meanOfY = np.mean(self.y)
        summation = np.sum(np.square((self.y - np.full((self.nrows,1),meanOfY))))
        return float(summation)


    def trainDriver(self,total_epochs,initial_learning_rate=0.001,final_learning_rate=0.011
                    ,learning_rate_increment_step=0.001,output_file_name='part1_readings.txt'):
        """
        Function to drive the train function. This function takes initital epoch and runs the train functio for given epoch
        After training the epoch value is reduced by 20% and again the train function is called.
        For every epoch value the train function is called for values starting from 
        given initial_learning_rate till final_learning_rate with increment of learning_rate_increment_step
        :param total_epochs: The total number of epoch for which the train function will be called
        :param initial_learning_rate: initial learning rate for training driver
        :param final_learning_rate: final learning rate for training driver, must be less than initial_learning_rate
        :param learning_rate_increment_step: learning rate increment value
        :param output_file_name: name file to contain the readings of the training operation
        :return: Generates a file with name given in output_file_name
        """
        totalEpochs = total_epochs
        leariningRate = initial_learning_rate
        count = 1;
        with open(output_file_name, 'a') as the_file:
            # print("Index \t Epochs \t Learning Rate \t MSE Training Data \t MSE Testing data")
            the_file.write("Index,Epochs,Learning Rate,MSE Training Data,MSE Testing data")
            while totalEpochs > 0 :
                while leariningRate<final_learning_rate:
                    #model.W = np.random.rand(self.ncols - 1).reshape(self.ncols - 1, 1)
                    #print("Weight values : ",self.W)
                    self.W = globalW
                    W, e, i, trainmse = self.train(epochs=totalEpochs,learning_rate=leariningRate)
                    testmse, rSquare = self.predict("test.csv")
                    # print(count,"\t",totalEpochs,"\t",leariningRate,"\t",trainmse,"\t",testmse.flatten()[0])
                    the_file.write("\n"+str(count)+","+str(totalEpochs)+","+str(leariningRate)+","+str(trainmse)+","+str(testmse.flatten()[0]))
                    leariningRate = float(leariningRate+learning_rate_increment_step)
                    count += 1
                totalEpochs = int(totalEpochs - 0.2*totalEpochs)
                leariningRate = initial_learning_rate


    def trainAndTestModelUsingLibrary(self,max_iterations,learning_rate,file_name):
        """
        This method performs the part 2 of the assignment. This method reads data from csv 
        and uses SDGRegressor from sklearn to train and then predict the output.
        :param max_iterations: Maximum iterations input for the SDG Regressor
        :param learning_rate: Learning rate
        :param file_name: Name of file to get input data. This method assumes that the data is already preprocessed.
        :return: Mean square error for the prediction
        """
        testDF = pd.read_csv(file_name)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        # traing using sgd
        clf = lm.SGDRegressor(max_iter=max_iterations, eta0=learning_rate, learning_rate='constant')
        # self.W = globalW
        clf.fit(self.X, self.y.flatten())
        outputY = clf.predict(testX)
        print("Mean Square Error for library : ",np.sqrt(err(testY,outputY)))
        return np.sqrt(err(testY,outputY))

    def driverForTrainUsingLibrary(self,total_epochs,initial_learning_rate=0.001,final_learning_rate=0.011
                    ,learning_rate_increment_step=0.001,output_file_name='part2_readings.txt'):
        """
        Function to drive the train function. This function takes initital epoch and runs the train functio for given epoch
        After training the epoch value is reduced by 20% and again the train function is called.
        For every epoch value the train function is called for values starting from 
        given initial_learning_rate till final_learning_rate with increment of learning_rate_increment_step
        :param total_epochs: The total number of epoch for which the train function will be called
        :param initial_learning_rate: initial learning rate for training driver
        :param final_learning_rate: final learning rate for training driver, must be less than initial_learning_rate
        :param learning_rate_increment_step: learning rate increment value
        :param output_file_name: name file to contain the readings of the training operation
        :return: Generates a file with name given in output_file_name
        """
        totalEpochs = total_epochs
        leariningRate = initial_learning_rate
        count = 1;
        with open(output_file_name, 'a') as the_file:
            # print("Index \t Epochs \t Learning Rate \t MSE Training Data \t MSE Testing data")
            the_file.write("Index,Epochs,Learning Rate,MSE Training Data,MSE Testing data")
            while totalEpochs > 0:
                while leariningRate < final_learning_rate:
                    testmse = self.partTwo(totalEpochs,leariningRate,"test.csv")
                    trainmse = self.partTwo(totalEpochs, leariningRate, "train.csv")
                    # the_file.write("\n" + str(count) + "," + str(totalEpochs) + "," + str(leariningRate) + "," + str(
                    #     trainmse) + "," + str(testmse.flatten()[0]))
                    the_file.write("\n" + str(count) + "," + str(totalEpochs) + "," + str(leariningRate) + "," + str(
                        trainmse) + "," + str(testmse.flatten()[0]))
                    leariningRate = float(leariningRate + learning_rate_increment_step)
                    count += 1
                totalEpochs = int(totalEpochs - 0.2 * totalEpochs)
                leariningRate = initial_learning_rate


if __name__ == "__main__":
    do_plot_graph = False
    if len(sys.argv) == 3:
        do_plot_graph = (sys.argv[1] == 'True')
        dataset_file_name = sys.argv[2]
        model = LinearRegression(data_set_file_name=dataset_file_name)
    elif len(sys.argv) == 2:
        do_plot_graph = (sys.argv[1] == 'True')
        model = LinearRegression()
    else:
        model = LinearRegression()
    globalW = model.W
    print("-------------------------- Part 1 Output -----------------")
    W, e, i, tmse = model.train(epochs=100, learning_rate=0.008,do_plot_graph=do_plot_graph)
    testmse, rSquare = model.predict("test.csv")
    print("Mean Square Error :",testmse)
    print("RSquare value :",rSquare)
    print("Final Weight values :\n",W)
    print("-------------------------- Part 1 Ends -------------------")
    print("-------------------------- Part 2 Output -----------------")
    model.trainAndTestModelUsingLibrary(100,0.008,"train.csv")
    print("-------------------------- Part 2 Ends -----------------")
