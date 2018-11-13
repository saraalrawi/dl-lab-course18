#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 8 19:12:24 2018

@author: sarajamal
"""

from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


# Create a interactive session
    '''
    The only difference between Session and an InteractiveSession is that InteractiveSession makes 
    itself the default session so that we can call run() or eval() without explicitly calling the session.
    '''
sess = tf.InteractiveSession()
# create place holders None for the batch_size
x_hold = tf.placeholder(tf.float32, [None, 28,28,1], name='x')
y_hold = tf.placeholder(tf.float32, [None, 10], name='y_')
y_pred = tf.placeholder(tf.float32, [None, 10])

def cnn_model_fn(x, lr, num_filters, filter_size):
    
    #Layers defined using Layers API
    # Conv Layer 1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=[filter_size, filter_size],
        padding="same",
        activation=tf.nn.relu)

    # Pooling 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Conv Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=num_filters,
        kernel_size=[filter_size, filter_size],
        padding="same",
        activation=tf.nn.relu)

    # Pooling 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Because the Dense layer is not 4 D we need to flatten the tensor
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * num_filters])

    # Dense Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Logits layer
    prediction = tf.layers.dense(inputs=dense, units=10)
    return prediction 


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters

    n_samples = x_train.shape[0]
    n_batches = n_samples // batch_size
    
    
    y_pred = cnn_model_fn(x_hold, lr, num_filters, filter_size)
    sess.run(tf.global_variables_initializer())
    
    # Loss, error between prediction and target
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_hold))
    
    # GDOptimizer tries to minmize the loss
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

 
    prediction_c = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_hold,1))
    
    # determine accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction_c, tf.float32), name="accuracy")
    
    saver = tf.train.Saver()

    learning_curve_c  = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        for b in range(n_batches):
            # create batches copy from last exercise
            x_batch = x_train[b * batch_size: (b + 1) * batch_size]
            y_batch = y_train[b * batch_size: (b + 1 ) * batch_size]
            
            train_step.run(feed_dict={x_hold: x_batch, y_hold: y_batch})
            
        # Evaluate the model 
        train_accuracy = accuracy.eval(feed_dict={x_hold: x_train, y_hold: y_train})
        # Objective to get the loss 
        learning_curve_c[epoch] = 1 - accuracy.eval(feed_dict={x_hold:x_valid, y_hold:y_valid})
        print("epoch %d, training accuracy %g"%(epoch + 1, train_accuracy))
    path_to_model = saver.save(sess, '/Users/sarajamal/Documents/MSc. Computer Science - Freiburg/WS2018:2019/WS1819/DL LAB/dl-lab-2018/exercise2/myModel.ckpt')
    print("Model saved in path: %s" % path_to_model)
    return learning_curve_c, path_to_model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    # using restore we are gonig to restore the graph and the weights
    saver.restore(sess, model)
    accuracy = graph.get_tensor_by_name("accuracy:0")
    x_hold = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    test_error = accuracy.eval(feed_dict={x_hold: x_test, y_: y_test})
    
    print('The test Error:' , test_error)
    return test_error

def plot_loss_graph(learning_curve_container , hyper_value, num_epochs , hpname):
    for i in range(len(learning_rate)):
        epoch = [k for k in range(num_epochs)]
        plt.plot(epoch, np.asarray(learning_curve_container[i]), label =str(hyper_value[i]))
        plt.xlabel('#Epochs')
        plt.ylabel('learning curve')
        plt.legend(loc='upper right')
    plt.show
    plt.savefig('loss_'+ hpname +'.png') 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=10, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = 3
    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve.tolist()
    results["test_error"] = test_error.tolist()

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()

    #Tryout different learning rates and save the results in results_learning_rates
    learning_rate = [0.1, 0.01, 0.001, 0.0001] #
    container = np.zeros([len(learning_rate), epochs]) 
    for i in range(len(learning_rate)):
        
        print("Hyperparametet learning_rate:" , learning_rate[i] )
        x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, learning_rate[i], num_filters, batch_size, filter_size)
        
        test_error = test(x_test, y_test, model)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["lr"] = learning_rate[i]
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()
        container[i]=results["learning_curve"]
        
        

        path = os.path.join(args.output_path, "results_learning_rates")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_run_" + str(i+1) + ".json")

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()
    plot_loss_graph(container , learning_rate, epochs , 'learing_rate')
    # Tryout different filters, and save the result in results_filters
    filters = [1, 3, 5, 7]
    container_filters = np.zeros([len(filters), epochs]) 
    for i in range(len(filters)):
        
        print("Hyperparametet Filter size:" , filters[i] )
        x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filters[i])
        
        # Using the saved model (e.g. graph and weights, do test!)
        test_error = test(x_test, y_test, model)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["lr"] = lr
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()

        path = os.path.join(args.output_path, "results_filters")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_run_" + str(i+1) + ".json")
        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()
    plot_loss_graph(container_filters , filters, epochs , 'filters')

