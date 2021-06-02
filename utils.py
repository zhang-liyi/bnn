import tensorflow as tf
import numpy as np

def load_mnist():
    return tf.keras.datasets.mnist.load_data(path='mnist.npz')


def prepare_data(x_train, y_train, x_test, y_test, method='vi', batch_size=128):
    
    train_sz = int(x_train.shape[0] *.8 //1)
    test_sz  = x_train.shape[0] - train_sz
    
    n_classes = max(y_train)+1

    if method == 'vi':
        x_train, x_val = tf.split(x_train, [train_sz, test_sz], axis=0)
        y_train, y_val = tf.split(y_train, [train_sz, test_sz], axis=0)
        
        y_train = tf.one_hot(y_train, n_classes)/1
        y_val = tf.one_hot(y_val, n_classes)/1
        y_test = tf.one_hot(y_test, n_classes)/1
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        test_dataset = None

    elif method == 'mcd':
        x_train, x_val = tf.split(x_train, [train_sz, test_sz], axis=0)
        y_train, y_val = tf.split(y_train, [train_sz, test_sz], axis=0)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
        	(x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
        	(x_test, y_test)).batch(batch_size)
        	
    elif method == 'hmc':
        x_train, x_val = tf.split(x_train, [train_sz, test_sz], axis=0)
        y_train, y_val = tf.split(y_train, [train_sz, test_sz], axis=0)
        
        train_dataset = None
        test_dataset = None

    return x_train, y_train, x_val, y_val, x_test, y_test, train_dataset, test_dataset

def make_2d_data(x_lim, y_lim, step):
    x1 = x_lim[0]
    x2 = x_lim[1]
    y1 = y_lim[0]
    y2 = y_lim[1]
    X = np.arange(x1, x2, step)
    Y = np.arange(y1, y2, step)
    X, Y = np.meshgrid(X, Y)
    data = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            data.append([X[i,j], Y[i,j]])
    data = np.array(data)[:,:,None]
    return data, X, Y

def print_status(method='method'):
	print()
	print('=============================')
	print('Running ' + method)
	print('=============================')






