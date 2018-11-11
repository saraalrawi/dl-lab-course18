import logging

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import argparse

from cnn_mnist import mnist, cnn_model_fn

logging.basicConfig(level=logging.WARNING)

import hpbandster.visualization as hpvis
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.
        rgs:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        epochs = budget
        filter_size = config['filter_size']
        # TODO: train and validate your convolutional neural networks here

        n_samples = self.x_train.shape[0]
        n_batches = n_samples // batch_size
        sess = tf.InteractiveSession()
        # Placeholder for x and y
        x_hold = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        y_hold = tf.placeholder(tf.float32, [None, 10], name='y_')
        #Placeholder for the predictions
        y_pred = tf.placeholder(tf.float32, [None, 10])
        
        # call the model and produce prediction

        y_pred = cnn_model_fn(x_hold, lr, num_filters, filter_size)
        sess.run(tf.global_variables_initializer())
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_hold))
        
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
        
        prediction_c = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_hold,1))
        accuracy = tf.reduce_mean(tf.cast(prediction_c, tf.float32), name="accuracy")
    
       
        for epoch in range(epochs):

            for b in range(n_batches):
                x_batch = self.x_train[b * batch_size:(b + 1) * batch_size]
                y_batch = self.y_train[b * batch_size:(b + 1) * batch_size]


                train_step.run(feed_dict={x_hold: x_batch, y_hold: y_batch})

            train_accuracy = accuracy.eval(feed_dict={x_hold: self.x_train, y_hold: self.y_train})
            validation_error = 1 - accuracy.eval(feed_dict={x_hold: self.x_valid, y_hold: self.y_valid})
            # epoch + 1 to fix index
            print("step %d, training accuracy %g" % (epoch +1, train_accuracy))
            
            info = 'First try'

        # TODO: We minimize so make sure you return the validation error here
        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': info # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():

        config_space = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1e-1, default_value='1e-2', log=True)

        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=64, log=True)

        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=16, log=True)
        filter_size = CSH.CategoricalHyperparameter('filter_size', ['3', '4', '5'])

        config_space.add_hyperparameters([lr, batch_size, num_filters, filter_size])

        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,


hpvis.losses_over_time(all_runs)


plt.savefig("random_search.png")

# TODO: retrain the best configuration (called incumbent) and compute the test error