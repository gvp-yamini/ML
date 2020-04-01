#####################################################################################################################
#   CS 6375 - Assignment 2, Neural Networks
#   Assignment performed by
#   Ameya Kulkarni  (ANK190006)
#   Yamini Thota    (YRT190003)
#
#####################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

"""
Class DataPreprocessor had generic methods to preprocess the ndarrays

"""
class DataPreprocessor:

    def scale_using_min_max(self,data_array):
        """
        Normalization
        This function scales the values of the data_array using the min_max algorithm
        X = (X - Xmin) / (Xmax - Xmin)
        This function changes values in the given array itself
        :param data_array: ndarray to be scaled
        :return: ndarray normalized data array
        """
        x_min = np.min(data_array)
        x_max = np.max(data_array)
        diff_x_max_x_min = x_max - x_min
        for index in range(data_array.shape[0]):
            data_array[index] = (data_array[index] - x_min)/diff_x_max_x_min
        return data_array

    def scale_using_mean(self,data_array):
        """
        Standardization
        This function scales the values using the mean and standard deviation
        X = (X - mean) / standard_deviation 
        :param data_array: ndarray to be scaled
        :return: ndarray standardized data array
        """
        data_array_mean = np.mean(data_array)
        data_array_standard_deviation = np.std(data_array)
        for index in range(data_array.shape[0]):
            data_array[index] = (data_array[index] - data_array_mean)/data_array_standard_deviation
        return data_array

    def preprocess_data(self,data,feature_labels):
        """
        Method to preprocess the data
        :param data: Input data frame
        :return: preprocessed data
        """
        data[:, feature_labels['alcohol']] = self.scale_using_min_max(data[:, feature_labels['alcohol']])
        data[:, feature_labels['pH']] = self.scale_using_min_max(data[:, feature_labels['pH']])
        data[:, feature_labels['quality']] = self.scale_using_min_max(data[:, feature_labels['quality']])
        data[:, feature_labels['residual sugar']] = self.scale_using_min_max(
            self.scale_using_mean(data[:, feature_labels['residual sugar']]))
        data[:, feature_labels['total sulfur dioxide']] = self.scale_using_min_max(
            self.scale_using_mean(data[:, feature_labels['total sulfur dioxide']]))
        return data


"""
This class represents the actual neural network and contains the methods for training the network and testing the network
This class will also contain methods to evaluate various functions like Sigmoid, Relu, Tanh function.
Following is the design of the network chosen
Number of input features    :   7

"""
class NeuralNet:
    def __init__(self):
        np.random.seed(1)
        self.W1 = np.random.rand(8,3)
        self.W2 = np.random.rand(4,2)
        self.W3 = np.random.rand(3,1)

    def activation_sigmoid(self,x):
        """
        This function returns the values after applying sigmoid function
        :param x: input value
        :return: sigmoid(x)
        """
        sig =  1/(1+np.exp(-x,dtype=np.float64))
        sig = np.minimum(sig, 0.9999)  # Set upper bound
        sig = np.maximum(sig, 0.0001)  # Set lower bound
        return sig

    def activation_tanh(self,x):
        """
        This function returns the value after applying tanh
        :param x: input value
        :return: tanh(x)
        """
        # tanh_value = (np.exp(x,dtype=np.float128)-np.exp(-x,dtype=np.float128))/(np.exp(x,dtype=np.float128)+np.exp(-x,dtype=np.float128))
        # we were getting Nan for tanh and np.exp was goin out of bounds hence we used tanh in terms of sigmoid.
        tanh_value = (2*self.activation_sigmoid(2*x)) - 1
        return tanh_value

    def activation_relu(self,x):
        """
        This function returns the value after applying ReLu function
        :param x: input value
        :return: ReLu(x)
        """
        return np.maximum(0,x)

    def forward_pass(self,X,activation_fun_name):
        """
        Function to compute the Forward pass in the network
        :param X: The input feature matrix
        :param activation_fun_name: The activation function to be used
        :return: Output value i.e. predicted Y value
        """
        net_layer1 = np.dot(X, self.W1)
        h = self.apply_activation_function(activation_fun_name,net_layer1)
        # add bias term
        net_layer1_with_bias = np.insert(h,0,1,axis=1)
        net_layer2 = np.dot(net_layer1_with_bias,self.W2)
        h = self.apply_activation_function(activation_fun_name,net_layer2)
        # add bias term
        net_layer2_with_bias = np.insert(h, 0, 1, axis=1)
        net_layer3 = np.dot(net_layer2_with_bias, self.W3)
        return net_layer1_with_bias,net_layer2_with_bias, self.apply_activation_function(activation_fun_name,net_layer3)


    def apply_activation_function(self,activation_fun_name,net):
        """
        Function to return value after applying activation function
        :param activation_fun_name: Name of Activation function to apply
        :param net: input value (in formule it is net i.e. summation(wixi) )
        :return: activationfunction(net) value
        """
        if activation_fun_name.lower() == "sigmoid".lower() :
            return self.activation_sigmoid(net)
        elif activation_fun_name.lower() == "tanh".lower() :
            return self.activation_tanh(net)
        elif activation_fun_name.lower() == "ReLu".lower() :
            return self.activation_relu(net)
        else:
            raise ValueError("Please specify proper activation function, Activation function can have values "
                             "'sigmoid', 'tanh' or 'relu'")



    def calculate_delta_for_output_weight(self,error,net,activation_fun_name):
        if activation_fun_name.lower() == "sigmoid".lower() :
            return self.calculate_delta_for_output_weight_sigmoid(error,net)
        elif activation_fun_name.lower() == "tanh".lower() :
            return self.calculate_delta_for_output_weight_tanh(error,net)
        elif activation_fun_name.lower() == "ReLu".lower() :
            return self.calculate_delta_for_output_weight_relu(error,net)

    def calculate_delta_for_output_weight_sigmoid(self,error,net):
        #print("Error :",error[0][0]," net ,",net[0][0])
        delta_output = np.multiply(np.multiply(net,(1-net),dtype=np.float64),error)
        #print(delta_output)
        return delta_output

    def calculate_delta_for_output_weight_tanh(self,error,net):
        #print("Error :",error[0][0]," net ,",net[0][0])
        # delta_output = np.multiply(np.square(net,dtype=np.float64),error)
        delta_output = np.multiply(1 - np.square(net,dtype=np.float64),error)
        #print(delta_output)
        return delta_output

    def calculate_delta_for_output_weight_relu(self,error,net):
        #print("Error :",error[0][0]," net ,",net[0][0])
        return np.where(net<=0,0,error)
        #print(delta_output)
        # return delta_output


    def train(self,data,testdata,epoch=1000,learningrate=0.00001,activation_function='sigmoid',do_plot_graph = False):

        nrows, ncols = data.shape[0], data.shape[1]
        X = data[:, 0:(ncols - 1)]
        # print(X.shape)
        y = data[:, (ncols - 1)].reshape(nrows, 1)
        # print(y.shape)

        nrows_test, ncols_test = testdata.shape[0], testdata.shape[1]
        X_test = test_data[:, 0:(ncols_test - 1)]
        # print(X_test.shape)
        y_test = test_data[:, (ncols_test - 1)].reshape(nrows_test, 1)
        # print(y_test.shape)

        mse = np.empty([1])
        iterations = np.empty([1])
        for count in range(epoch):
            #--print("------------------------------ running for count : ",count,"-------------------")
            net_layer1, net_layer2, pred = self.forward_pass(X, activation_function)
            #--print("shapes : net1 : ", net_layer1.shape, "\t net2 :", net_layer2.shape, "\t prediction :", pred.shape)
            # --------------------------------------- updating Weight for W3 --------------------------------------------
            error = y - pred # tj - oj
            # print np.dot(error.T,error)
            mean_square_error = 0.5 * np.dot(np.transpose(error), error)
            #--print("Error : ", error.shape, ", mean_square_error : ", mean_square_error)


            #----------------------------------- considering tanh -------------------------
            # delta_for_output_layer = self.calculate_delta_for_output_weight_relu(error, pred)
            delta_for_output_layer = self.calculate_delta_for_output_weight(error, pred,activation_function)
            #--print("Delta for output : ", delta_for_output_layer.shape)
            dw1, dw2, dw3 = self.backward_pass(X, delta_for_output_layer, learningrate
                                               , net_layer1, net_layer2,activation_function)
            #------------------------------------------------------------------------------------

            self.W1 = self.W1 + dw1
            self.W2 = self.W2 + dw2
            self.W3 = self.W3 + dw3
            #print("W1 new : ", self.W1)
            #print("W2 new : ", self.W2)
            #print("W3 new : ", self.W3)
            mse = np.append(mse,mean_square_error)
            iterations = np.append(iterations,count)


        if do_plot_graph:
            iterations = np.delete(iterations, 0)
            mse = np.delete(mse, 0)
            plt.plot(iterations, mse, linestyle='-', marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Mean Square Error")
            plt.show()


        #--print("-------------------------------- With new weights -----------------------")
        net_layer1_train, net_layer2_train, pred_train = self.forward_pass(X, activation_function)
        train_error = y - pred_train
        # print np.dot(error.T,error)
        mean_square_error_train = 0.5 * np.dot(np.transpose(train_error), train_error)
        net_layer1_test, net_layer2_test, pred_test = self.forward_pass(X_test, activation_function)
        test_error = y_test - pred_test
        # print np.dot(error.T,error)
        mean_square_error_test = 0.5 * np.dot(np.transpose(test_error), test_error)

        print("Mean_square_error_train : ", mean_square_error_train, ", Mean_square_error_test : ", mean_square_error_test)
        return mean_square_error_train,mean_square_error_test



    def backward_pass(self, X, delta_for_output_layer, learningrate, net_layer1, net_layer2,activation_function):
        # calculate delta for previous layer
        delta_for_W2 = np.dot(self.W3, delta_for_output_layer.T)
        #print(delta_for_W2.shape)
        delta_for_W2 = np.sum(delta_for_W2, axis=1).reshape(3, 1)
        #print(delta_for_W2.shape)
        net_output_of_all_inputs = []
        if activation_function.lower() == "sigmoid".lower() :
            net_output_of_all_inputs = np.sum(np.multiply(net_layer2, (1 - net_layer2), dtype=np.float64),
                                              axis=0).reshape(3, 1)
        elif activation_function.lower() == "tanh".lower() :
            net_output_of_all_inputs = np.sum((1 - np.square(net_layer2, dtype=np.float64)), axis=0).reshape(3, 1)
        elif activation_function.lower() == "ReLu".lower() :
            #TODO - impl relu
            net_output_of_all_inputs = np.sum(np.where(net_layer2<=0,0,net_layer2), axis=0).reshape(3, 1)
        # -- sigmoid prime  net_output_of_all_inputs = np.sum(np.multiply(net_layer2, (1 - net_layer2),dtype=np.float64), axis=0).reshape(3, 1)
        # -- tanh prime     net_output_of_all_inputs = np.sum((1 - np.square(net_layer2,dtype=np.float64)), axis=0).reshape(3, 1)
        #-print("Net2 * (1-net2) :", net_output_of_all_inputs)
        delta_for_layer_2 = np.multiply(net_output_of_all_inputs, delta_for_W2,dtype=np.float64)
        #-print("Delta_for_layer_2 : ", delta_for_layer_2.shape)
        # delta_output = np.multiply(np.multiply(net_layer2, (1 - net_layer2)), error)
        # delta_for_W1 = np.dot(self.W2,delta_for_layer_2.T)
        delta_for_layer_2_omitting_row0 = np.delete(delta_for_layer_2, (0), axis=0)
        delta_for_W1 = np.dot(self.W2, delta_for_layer_2_omitting_row0)
        # print(delta_for_W1)
        # print(delta_for_W1.shape)
        net1_output_of_all_inputs = []
        if activation_function.lower() == "sigmoid".lower():
            net1_output_of_all_inputs = np.sum(np.multiply(net_layer1, (1 - net_layer1), dtype=np.float64),
                                               axis=0).reshape(4, 1)
        elif activation_function.lower() == "tanh".lower():
            net1_output_of_all_inputs = np.sum((1 - np.square(net_layer1, dtype=np.float64)), axis=0).reshape(4, 1)
        elif activation_function.lower() == "ReLu".lower():
            # TODO - impl relu
            net1_output_of_all_inputs = np.sum(np.where(net_layer1 <= 0, 0, net_layer1), axis=0).reshape(4, 1)
        # -- sigmoid prime  net1_output_of_all_inputs = np.sum(np.multiply(net_layer1, (1 - net_layer1),dtype=np.float64), axis=0).reshape(4, 1)
        # -- tanh prime     net1_output_of_all_inputs = np.sum((1 - np.square(net_layer1,dtype=np.float64)), axis=0).reshape(4, 1)
        #-print("Net1 * (1-net1) :", net1_output_of_all_inputs)
        delta_for_layer_1 = np.multiply(net1_output_of_all_inputs, delta_for_W1,dtype=np.float64)
        # print("Delta_for_layer_1 : ", delta_for_layer_1)
        delta_for_layer_1_omitting_row0 = np.delete(delta_for_layer_1, (0), axis=0)
        #-print("Delta_for_layer_1 : ", delta_for_layer_1_omitting_row0)
        # sum of delta values for all data inputs


        #---------------------------------------------------
        # update all weights
        # update W3
        net_layer2[:, 0] = np.multiply(net_layer2[:, 0].reshape(net_layer2.shape[0], 1),
                                       delta_for_output_layer).flatten()
        net_layer2[:, 1] = np.multiply(net_layer2[:, 1].reshape(net_layer2.shape[0], 1),
                                       delta_for_output_layer).flatten()
        net_layer2[:, 2] = np.multiply(net_layer2[:, 2].reshape(net_layer2.shape[0], 1),
                                       delta_for_output_layer).flatten()
        delta_W_for_W3_all_inputs = net_layer2 * learningrate
        #-print("Delta W for W3 :", delta_W_for_W3_all_inputs.shape)
        # sum of delta values for all data inputs
        delta_W_for_W3 = np.sum(delta_W_for_W3_all_inputs, axis=0).reshape(3, 1)
        #-print("Delta W for W3 :", delta_W_for_W3)
        # update W3
        #-print("W3 old : ", self.W3)
        #--self.W3 = self.W3 + delta_W_for_W3
        # print("W3 new : ", self.W3)
        # update W2
        #-print("W2 old : ", self.W2)
        net_layer1 = np.sum(net_layer1, axis=0).reshape(4, 1)
        #-print("Net layer 1:", net_layer1)
        #-print("Delta from layer 2 :", delta_for_layer_2_omitting_row0)
        product_of_net_and_delta2 = np.dot(net_layer1, delta_for_layer_2_omitting_row0.T)
        #-print(product_of_net_and_delta2)
        DELTA_W2 = product_of_net_and_delta2 * learningrate
        #-print(DELTA_W2)
        #--self.W2 = self.W2 + DELTA_W2
        #print("W2 new : ", self.W2)
        # update W1
        #-print("W1 old : ", self.W1)
        net_layer0 = np.sum(X, axis=0).reshape(8, 1)
        #-print("Net layer 1:", net_layer0)
        #-print("Delta from layer 2 :", delta_for_layer_1_omitting_row0)
        product_of_net_and_delta1 = np.dot(net_layer0, delta_for_layer_1_omitting_row0.T)
        #-print(product_of_net_and_delta1)
        DELTA_W1 = product_of_net_and_delta1 * learningrate
        #-print(DELTA_W1)
        #--self.W1 = self.W1 + DELTA_W1
        #print("W1 new : ", self.W1)
        return DELTA_W1,DELTA_W2,delta_W_for_W3


    def trainDriver(self,data,testdata,total_epochs,initial_learning_rate=0.001,final_learning_rate=0.011
                    ,learning_rate_increment_step=0.001,output_file_name='readings.txt'):
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
        activation_functions = ['sigmoid','tanh','relu']
        with open(output_file_name, 'a') as the_file:
            # print("Index \t Epochs \t Learning Rate \t MSE Training Data \t MSE Testing data")
            the_file.write("Index,Activation,Epochs,Learning Rate,MSE Training Data,MSE Testing data")
            for activation in activation_functions:
                totalEpochs = total_epochs
                while totalEpochs > 0 :
                    while leariningRate<final_learning_rate:
                        self.W1 = globalW1
                        self.W2 = globalW2
                        self.W3 = globalW3
                        trainmse,testmse = self.train(data=data,testdata=testdata,epoch=totalEpochs,learningrate=leariningRate,activation_function=activation)
                        the_file.write("\n"+str(count)+","+activation+","+str(totalEpochs)+","+str(leariningRate)+","+str(trainmse.flatten()[0])+","+str(testmse.flatten()[0]))
                        leariningRate = float(leariningRate+learning_rate_increment_step)
                        count += 1
                    totalEpochs = int(totalEpochs - 0.2*totalEpochs)
                    leariningRate = initial_learning_rate


if __name__ == "__main__":

    activation_function_name = None
    dataset_file_name = 'winequality-red.csv'
    do_plot_graph = False
    if len(sys.argv) == 4:
        activation_function_name = str(sys.argv[1])
        do_plot_graph = (sys.argv[2] == 'True')
        dataset_file_name = str(sys.argv[3])

    #   Load data from csv file
    df = pd.read_csv(filepath_or_buffer=dataset_file_name,delimiter=';')

    # --------------------------------- Data Analysis ------------------------------------
    print(df.describe())

    #   Find correlations between features and output those that have correlation greater than 0.65
    correlations = df.corr()
    cols = df.columns
    print(correlations)
    print("Feature 1\t | Feature 2\t | Correlation")
    for i in range(0,correlations.shape[0]):
        for j in range(0,correlations.shape[1]):
            if (float(correlations.iloc[i][j]) > 0.65 or float(correlations.iloc[i][j]) < -0.65) and i!=j:
                print(cols[i]," | ",cols[j]," \t | ",correlations.iloc[i][j])

    # ---------------------------------------------------------------------------------------
    #   drop features based on correlations (Reason to drop these columns is mentioned in the document)
    columns_to_drop = ['citric acid','free sulfur dioxide','fixed acidity']
    df = df.drop(columns=columns_to_drop)
    # ---------------------------------------------------------------------------------------

    #   make dictionary for feature labels to use for manipulating the data columns
    feature_labels = {}
    columns = df.columns
    for i in range(len(columns)):
        feature_labels[columns[i]] = i
    print("Feature Labels --> ", feature_labels)


    # convert dataframe to ndarray
    data = df.iloc[:,:].values.reshape(df.shape[0], df.shape[1])

    # preprocess the data
    preProcessor = DataPreprocessor()
    data = preProcessor.preprocess_data(data,feature_labels)

    # split dataset
    indices = np.random.permutation(data.shape[0])
    training_indices, test_indices = indices[:int((data.shape[0]*0.8))], indices[int((data.shape[0]*0.8)):]
    training_data, test_data = data[training_indices, :], data[test_indices, :]

    nn = NeuralNet()
    globalW1 = nn.W1
    globalW2 = nn.W2
    globalW3 = nn.W3
    if activation_function_name is not None:
        nn.train(data=training_data,testdata=test_data,activation_function=activation_function_name,do_plot_graph=do_plot_graph)
    else:
        nn.train(data=training_data, testdata=test_data)
    # nn.trainDriver(data=training_data,testdata=test_data,total_epochs=1000)
