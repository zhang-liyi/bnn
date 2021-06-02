import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Layer, Input, Flatten, Dense, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import activations, initializers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow import nn
import numpy as np

# %%
tf.keras.backend.set_floatx('float32')
tf.get_logger().setLevel('ERROR')

# %%
num_hidden = 20

# %% Variational Inference

class DenseVariational(Layer):

    def __init__(self,units,kl_weight,activation=None,
        prior_sigma_1=1.5,prior_sigma_2=0.1,prior_pi=0.5,**kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.TruncatedNormal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.TruncatedNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.Constant(-3.),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.Constant(-3.),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfd.Normal(mu, sigma)

        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfd.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfd.Normal(0.0, self.prior_sigma_2)

        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))


class VIModel(tf.keras.Model):

    def __init__(self, kl_weight, prior_params, output_size=2):
        super(VIModel, self).__init__()
        self.output_size = output_size
        self.kl_weight = kl_weight
        self.prior_params = prior_params
        self.flatten = Flatten()
        self.dense1 = DenseVariational(num_hidden, self.kl_weight, **self.prior_params, activation='relu')
        self.dense2 = DenseVariational(num_hidden, self.kl_weight, **self.prior_params, activation='relu')
        self.dense3 = DenseVariational(self.output_size, self.kl_weight, **self.prior_params, activation='softmax')

    def call(self, x_in):
        x = self.flatten(x_in)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class BNN_VI():

    def __init__(self, prior_params, output_size=2):
        self.prior_params = prior_params
        self.output_size = output_size

    def body_train_samples(self, loss_list, x_batch_train, y_batch_train, sample_size, i):
        logits = self.model(x_batch_train)
        loss_fn = CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        loss_value = loss_fn(y_batch_train, logits)
        loss_list = tf.concat([loss_list, [loss_value]], axis=0)
        i = i + 1
        return loss_list, x_batch_train, y_batch_train, sample_size, i

    def cond_train_samples(self, loss_list, x_batch_train, y_batch_train, sample_size, i):
        return i < sample_size

    def body_eval_samples(self, pred_list, x, eval_sample_size, i):
        logits = self.model(x)
        pred_value = nn.softmax(logits)
        pred_list = tf.concat([pred_list, [pred_value]], axis=0)
        i = i + 1
        return pred_list, x, eval_sample_size, i

    def cond_eval_samples(self, pred_list, x, eval_sample_size, i):
        return i < eval_sample_size
    
    def predict(self, x, sample_size=10):
        pred_list = tf.constant(tf.zeros((1, x.shape[0], self.output_size)))
        pred_list, x_, s_, i_ = tf.while_loop(self.cond_eval_samples,self.body_eval_samples, 
        	loop_vars=[pred_list, x, sample_size, 0],
        	shape_invariants=[tf.TensorShape([None, x.shape[0], self.output_size]),x.shape,tf.TensorShape([]),tf.TensorShape([])])
        pred_list = tf.gather(pred_list, list(range(1, sample_size+1)))
        return pred_list

    def train(self, x_train, y_train, x_val, y_val, train_dataset,
        learning_rate=1e-5, epochs=300, batch_size=128, sample_size=1, eval_sample_size=1):
        
        self.kl_weight = batch_size / x_train.shape[0]
        self.model = VIModel(self.kl_weight, self.prior_params)

        optimizer = SGD(lr=learning_rate, momentum=0.95)
        train_metric = CategoricalAccuracy()
        val_metric = CategoricalAccuracy()
        test_metric = CategoricalAccuracy()

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                
                # kl_reweight = 1 / self.kl_weight * (2**(num_batches-step-1)) / (2**num_batches - 1)
                kl_reweight = 1.

                with tf.GradientTape() as tape:
                    loss_list = tf.constant(tf.zeros(1))
                    loss_list, x_, y_, s_, i_ = tf.while_loop(self.cond_train_samples,
                                                              self.body_train_samples, 
                                                              loop_vars=[loss_list,
                                                                         x_batch_train, y_batch_train, 
                                                                         sample_size, 0],
                                                              shape_invariants=[tf.TensorShape([None]),  
                                                                                x_batch_train.shape,
                                                                                y_batch_train.shape,
                                                                                tf.TensorShape([]),
                                                                                tf.TensorShape([])])
                    loss_list = tf.gather(loss_list, list(range(1, sample_size+1)))
                    loss_value = tf.reduce_mean(loss_list, axis=0)
                    loss_value += sum(self.model.losses) * kl_reweight
                    loss_value = loss_value

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if step % 200 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))

            pred_list = tf.constant(tf.zeros((1, x_train.shape[0], y_train.shape[1])))
            pred_list, x_, s_, i_ = tf.while_loop(self.cond_eval_samples,
                                                  self.body_eval_samples, 
                                                  loop_vars=[pred_list, x_train, 
                                                             eval_sample_size, 0],
                                                  shape_invariants=[tf.TensorShape([None, x_train.shape[0], y_train.shape[1]]), 
                                                                    x_train.shape,
                                                                    tf.TensorShape([]),
                                                                    tf.TensorShape([])])
            pred_list = tf.gather(pred_list, list(range(1, eval_sample_size+1)))
            pred_value = tf.reduce_mean(pred_list, axis=0)
            
            train_metric.update_state(y_train, pred_value)
            train_acc = train_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            train_metric.reset_states()

            val_pred_list = tf.constant(tf.zeros((1, x_val.shape[0], y_val.shape[1])))
            val_pred_list, x_, s_, i_ = tf.while_loop(self.cond_eval_samples,
                                                      self.body_eval_samples, 
                                                      loop_vars=[val_pred_list, x_val, 
                                                                 eval_sample_size, 0],
                                                      shape_invariants=[tf.TensorShape([None, x_val.shape[0], y_val.shape[1]]), 
                                                                        x_val.shape,
                                                                        tf.TensorShape([]),
                                                                        tf.TensorShape([])])
            val_pred_list = tf.gather(val_pred_list, list(range(1, eval_sample_size+1)))
            val_pred_value = tf.reduce_mean(val_pred_list, axis=0)

            val_metric.update_state(y_val, val_pred_value)
            val_acc = val_metric.result()
            print("Validation acc over epoch: %.4f" % (float(val_acc),))
            val_metric.reset_states()

        return


# %% Monte Carlo Dropout

class MCDModel(Model):
    def __init__(self, output_size=10):
        super(MCDModel, self).__init__()
        self.output_size = output_size
        self.flatten = Flatten()
        self.dense1 = Dense(num_hidden, activation='relu')
        self.dropout1 = Dropout(0.5)
        # second layer
        self.dense2 = Dense(num_hidden, activation='relu')
        self.dropout2 = Dropout(0.5)
        # output
        self.dense3 = Dense(self.output_size, activation="softmax") 
    
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        
        return x

class BNN_MCD():

    def __init__(self, output_size=10):
        self.output_size = output_size
        self.model = MCDModel(output_size=output_size)
        
    def train_step(self, images, labels, model, optimizer, loss_object, train_loss, train_accuracy):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        train_loss(loss)
        
        train_accuracy(labels, predictions)


    def train(self, train_ds, learning_rate=1e-3, epochs=20):
        # define each metrics
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(epochs):
            for images, labels in train_ds:
                self.train_step(images, labels, self.model, optimizer, loss_object, train_loss, train_accuracy)

            template = "Epoch {}, Loss: {}, Accuracy: {}"
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100))

            # reset each metrics
            train_loss.reset_states()
            train_accuracy.reset_states()

    def body_eval_samples(self, pred_list, x, sample_size, i):
        pred_value = self.model(x, training=False)
        pred_list = tf.concat([pred_list, [pred_value]], axis=0)
        i = i + 1
        return pred_list, x, sample_size, i

    def cond_eval_samples(self, pred_list, x, sample_size, i):
        return i < sample_size

    def predict(self, data, sample_size=1):
        pred_list = tf.constant(tf.zeros((1, data.shape[0], self.output_size)))
        pred_list, x_, s_, i_ = tf.while_loop(self.cond_eval_samples,
                                              self.body_eval_samples, 
                                              loop_vars=[pred_list, data, 
                                                             sample_size, 0],
                                                  shape_invariants=[tf.TensorShape([None, data.shape[0], self.output_size]), 
                                                                    data.shape,
                                                                    tf.TensorShape([]),
                                                                    tf.TensorShape([])])
        pred_list = tf.gather(pred_list, list(range(1, sample_size+1)))
        return pred_list

            
            
# %% Hamiltonian Monte Carlo

class BNN_HMC():

    def __init__(self, prior_params, output_size=2):
        self.prior_params = prior_params
        self.prior_dist = tfd.Mixture(
            cat=tfd.Categorical(
                probs=[self.prior_params['prior_pi'], 1.-self.prior_params['prior_pi']]),
            components=[
            tfd.Normal(loc=0., scale=self.prior_params['prior_sigma_1']),
            tfd.Normal(loc=0., scale=self.prior_params['prior_sigma_2']),
            ])
        self.output_size = output_size

    def build_network(self, weight_list, bias_list, activation=tf.nn.relu):

        def model(x):
            x = Flatten()(x)
            for (weight, bias) in zip(weight_list[:-1], bias_list[:-1]):
                x = activation(tf.matmul(x, weight) + bias)
            x = tf.matmul(x, weight_list[-1]) + bias_list[-1]  # Final linear layer.

            return tfd.Categorical(logits=x)   # or build a trainable normal, etc.
    
        return model

    def log_joint_fn(self, weight1, bias1, weight2, bias2, weight3, bias3):
        weight_list = [weight1, weight2, weight3]
        bias_list = [bias1, bias2, bias3]

        # prior log-prob
        lp = sum([tf.reduce_sum(self.prior_dist.log_prob(weight)) for weight in weight_list])
        lp += sum([tf.reduce_sum(self.prior_dist.log_prob(bias)) for bias in bias_list])

        # likelihood of predicted labels
        network = self.build_network(weight_list, bias_list)
        labels_dist = network(self.x_train)
        lp += tf.reduce_sum(labels_dist.log_prob(self.y_train))

        return lp

    def posterior_predictive(self, x, posterior_samples):
        samples = [[Z[i] for Z in posterior_samples] for i in range(posterior_samples[0].shape[0])]

        predictive_samples = []
        for i in range(len(samples)):
            weight1, bias1, weight2, bias2, weight3, bias3 = samples[i]
            weight_list = [weight1, weight2, weight3]
            bias_list = [bias1, bias2, bias3]
            
            network = self.build_network(weight_list, bias_list)
            labels_dist = network(x)
            predictive_samples.append(labels_dist.sample().numpy())
            
        return predictive_samples

    def train(self, x_train, y_train, num_samp=5, step_size=6e-4, num_l_steps=20):
        self.x_train = x_train
        self.y_train = tf.cast(y_train, tf.int32)

        flattened_length = self.x_train.shape[1] * self.x_train.shape[2]
        weight_1 = self.prior_dist.sample([flattened_length, num_hidden])
        weight_2 = self.prior_dist.sample([num_hidden, num_hidden])
        weight_3 = self.prior_dist.sample([num_hidden, self.output_size])
        bias_1 = self.prior_dist.sample(num_hidden)
        bias_2 = self.prior_dist.sample(num_hidden)
        bias_3 = self.prior_dist.sample(self.output_size)
        current_state = [weight_1, bias_1, weight_2, bias_2, weight_3, bias_3]

        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_joint_fn,
            step_size=step_size,
            num_leapfrog_steps=num_l_steps)

        out = tfp.mcmc.sample_chain(
            num_samp, current_state, 
            previous_kernel_results=None, kernel=hmc_kernel,
            num_burnin_steps=10, num_steps_between_results=2, 
            trace_fn=(lambda current_state, kernel_results: kernel_results), 
            return_final_kernel_results=False,
            name=None)
        
        return out
