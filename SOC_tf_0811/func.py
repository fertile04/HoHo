import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


def get_data(NAME : str, drop_labels_x : list, drop_labels_y : list):
    """Reads .csv data and returns data(up to 2D) and data_y(1D) 

    Args:
        NAME (str): File location
        drop_labels_x (list): Labels to be dropped from original data (to form an input data)
        drop_labels_y (list): Labels to be dropped from original data (to form an output data)

    Returns:
        list, list: Input and output data before the sequence generation
    """
    data = pd.read_csv(NAME)
    data_y = data.copy()
    data = data.drop(drop_labels_x, axis = 1)
    data_y = data_y.drop(drop_labels_y, axis = 1)
    print(data.columns)
    print(data_y.columns)
    data = data.values
    data_y = data_y.values
    
    return data, data_y

def seq_gen_x(data_x, seq_len):
    """Gets input data and returns '+1 dimensional' datas divided by seq_len(sequence length).

    Args:
        data_x (list): Input data
        seq_len (int, optional): sequence length. Defaults to 5.

    Returns:
        np.array: sequence-divided input data
    """
    x_data = []
    for batch in range(data_x.shape[0]-seq_len-1):
        x_data.append(data_x[batch:batch + seq_len])
    x_data = np.array(x_data).astype(np.float32)
    
    return x_data

def seq_gen_y(data_y, seq_len):
    """Gets output data and returns '+1 dimensional' datas divided by seq_len(sequence length).

    Args:
        data_y (list): Output data
        seq_len (int, optional): sequence length. Defaults to 5.

    Returns:
        np.array: sequence-divided output data
    """

    y_data = []
    for batch in range(data_y.shape[0]-seq_len-1):
        y_data.append(data_y[batch+seq_len-1])
    y_data = np.array(y_data).astype(np.float32)
    
    return y_data

def split_data(x_data, y_data):
    
    split_len = int(round(x_data.shape[0] * 0.8))
    print(f'split_len = {split_len}')
    x_train = x_data[:, :, :split_len]
    y_train = y_data[:, :, :split_len]
    x_test = x_data[:, :, split_len:]
    y_test = y_data[:, :, split_len:]
    print(f'x_train = {x_train.shape}')
    print(f'y_train = {y_train.shape}')
    
    return x_train, y_train, x_test, y_test

def flatten_2Dto1D(data):
    data_flatten = data.reshape(int(data.shape[0] ), 1)
    
    return data_flatten

def prove(model, h5_path, x_data, y_data):
    model.load_weights(h5_path)
    prediction = model.predict(x_data)
    
    prediction_graph = flatten_2Dto1D(prediction)
    # y_graph = flatten_2Dto1D(y_data)
    y_graph = y_data
    
    Error_rate = []
    Error = []
    for step in range(len(prediction_graph)-100):
        Error_rate.append((prediction_graph[step] - y_graph[step]) / y_graph[step] * 100)
        Error.append(Error_rate[step] / 100)

    
    RMSE_total = np.sqrt(np.mean(np.square(Error)))
    MAE_total = np.mean(np.absolute(Error))

    return RMSE_total, MAE_total, Error_rate, prediction_graph, y_graph
        
def show_and_prove(model, h5_path, x_data, y_data, save_path, return_loss = False, show_y = True, plot = True):
    """Shows prediction and y data graphs. Also returns RMSE, MAE, and Error-by-steps.

    Args:
        model (tf.Model): Defined model
        h5_path (str) : .h5 file directory path
        x_data (np.array): Input data for the prediction
        y_data (np.array): Desired output data
        save_path (str) : Directory path to save the graph plots
        return_loss (bool, optional): return RMSE & MAE loss(list) if True. Defaults to False.
        show_y (bool, optional): y_data graph will be also plotted if True. Defaults to True.

    Returns:
        int, int, list: RMSE, MAE, Error rate by cycle steps.
    """
    # param = {'seq_len' : 25, 'sample_len' : 25, 'num_units' : 64, 'num_filters' : 64, 'window' : 3, 'drop_rate' : 0.2, 'num_epochs' : 800}
    RMSE_total, MAE_total, Error_rate, prediction_graph, y_graph = prove(model, h5_path, x_data, y_data)
    
    # print(prediction_graph.shape)
    # print(save_path)
    # if plot:
    #     pl.figure(dpi=350)
    #     pl.ylabel('Voltage')
    #     pl.xlabel('Time(s)')
    #     line = pl.plot(prediction_graph, label = 'Voltage Estimation')
    #     pl.setp(line, linewidth=0.5)
    #     if show_y:
    #         y_line = pl.plot(y_graph, label = 'Voltage Reference')
    #         pl.setp(y_line, linewidth=0.5)
    #     pl.legend()
    #     pl.savefig(f'{save_path}\Estimation.png')
    #     pl.show()
    
    if return_loss:
        return RMSE_total, MAE_total, Error_rate
#자신기준 뒤로
def lowpass(input, Fsize):
    output=[]


    Dsize=input.shape[0]
    for i in range (0,Dsize):
        sum=0
        for j in range(1-Fsize,1):
            sum+=input[i + j]/Fsize
        output.append(sum)
    output=np.array(output).astype(np.float32)    


    return output
#자신기준 양옆
def lowpass1(input, Fsize):
    output=[]
    size=int(Fsize/2)

    Dsize=input.shape[0]
    for i in range (0,Dsize-size-1):
        sum=0
        for j in range(-size,size+1):
            sum+=input[i+j]/Fsize
        output.append(sum)
    output=np.array(output).astype(np.float32)    


    return output

#3 필터계수 (0.25 0.5 0.25)
def lowpass2(input):
    output=[]

    Dsize=input.shape[0]
    for i in range (0,Dsize-1):
        sum=0.25*input[i-1]+0.5*input[i]+0.25*input[i+1]
        output.append(sum)
    output=np.array(output).astype(np.float32)    


    return output

#5 필터계수 (-0.125 0.25 0.75 0.25 -0.125)
def lowpass3(input):
    output=[]

    Dsize=input.shape[0]
    for i in range (0,Dsize-2):
        sum=0.25*input[i-1]+0.75*input[i]+0.25*input[i+1]-0.125*input[i+2]-0.125*input[i-2]
        output.append(sum)
    output=np.array(output).astype(np.float32)    


    return output


#자신기준 뒤로
def relow(refilter, real, Fsize):
    Rsize=refilter.shape[0]
    reoutput=[]
    for i in range (0,Rsize):
        sum=refilter[i]
        for j in range(1-Fsize,0):
            sum-= real[i+j]
        reoutput.append(sum/Fsize)
    reoutput=np.array(reoutput).astype(np.float32)


    return reoutput
#실제복호 자신기준 양옆
def relow1(refilter, real, Fsize):
    Rsize=refilter.shape[0]
    size=int(Fsize/2)
    reoutput=[]
    for i in range (0,Rsize-size-1):
        sum=Fsize*refilter[i]
        for j in range(-size,0):
            sum-=real[i+j]+real[i-j]
        reoutput.append(sum)
    reoutput=np.array(reoutput).astype(np.float32)    


    return reoutput

#핕터링 자신기준 양옆
def relow2(refilter, real, Fsize):
    Rsize=refilter.shape[0]
    size=int(Fsize/2)
    reoutput=[]
    for i in range (0,Rsize-size-1):
        sum=refilter[i]/Fsize
        for j in range(-size,0):
            sum+=(real[i+j]+real[i-j])/Fsize
        reoutput.append(sum)
    reoutput=np.array(reoutput).astype(np.float32)    


    return reoutput

#5/3 복호화 (-0.5 1 -0.5)
def relow3(input):
    output=[]

    Dsize=input.shape[0]
    for i in range (0,Dsize-1):
        sum=0.5*input[i-1]+input[i]+0.5*input[i+1]
        output.append(sum)
    output=np.array(output).astype(np.float32)    


    return output