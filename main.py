import simulations as S
import models as M
import utils as U

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
                        description='Bayesian Neural Network')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use.',
                        default='circles')
    parser.add_argument('--noise',
                       help='Noise of generating data',
                       type=float,
                       default=0.1)
    parser.add_argument('--testing',
                       default=True, 
                       type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()
    return args

args = parse_args()
save_dir = 'results/'+args.dataset+'/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

# %% Sample data
if args.dataset.lower() == 'circles':
	x_train, y_train, x_test, y_test = S.generate_circles(noise=args.noise, testing=args.testing)
	data, X, Y = U.make_2d_data((-1.5, 1.5), (-1.5, 1.5), 0.02)
if args.dataset.lower() == 'moons':
	x_train, y_train, x_test, y_test = S.generate_moons(noise=args.noise)
	data, X, Y = U.make_2d_data((-2.0, 3.0), (-1.5, 2.0), 0.02)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, s=3)
plt.savefig(save_dir+'training_data')
plt.close()


# %% Variational Inference
U.print_status('VI')
x_tr, y_tr, x_val, y_val, x_te, y_te, train_dataset, test_dataset = \
    U.prepare_data(x_train[:,:,None], y_train, x_test[:,:,None], y_test, method='vi')

prior_params = {
    'prior_sigma_1': np.float32(np.exp(0)), 
    'prior_sigma_2': np.float32(np.exp(-7)), 
    'prior_pi': 0.5 
}

bnn_vi_inst = M.BNN_VI(prior_params, output_size=2)
bnn_vi_inst.train(x_tr, y_tr, x_val, y_val, train_dataset,
                  learning_rate=1e-5, epochs=80, batch_size=128, sample_size=1, eval_sample_size=1)

y_pred = bnn_vi_inst.predict(data, sample_size=100).numpy()
y_pred = np.mean(y_pred, axis=0)
plt.scatter(X,Y,c=y_pred[:,0])
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='gray', s=3, alpha=0.2)
plt.savefig(save_dir+'map_vi')
plt.close()



# %% Monte Carlo Dropout
U.print_status('Monte Carlo Dropout')
x_tr, y_tr, x_val, y_val, x_te, y_te, train_dataset, test_dataset = \
    U.prepare_data(x_train, y_train, x_test, y_test, method='mcd')

bnn_mcd_inst = M.BNN_MCD(output_size=2)
bnn_mcd_inst.train(train_dataset, learning_rate=1e-3, epochs=80)

y_pred = bnn_mcd_inst.predict(data, sample_size=100).numpy()
y_pred = np.mean(y_pred, axis=0)
plt.scatter(X, Y, c=y_pred[:,0], s=3)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='gray', s=3, alpha=0.2)
plt.savefig(save_dir+'map_mcd')
plt.close()



# %% Hamiltonian Monte Carlo
U.print_status('HMC')
x_tr, y_tr, x_val, y_val, x_te, y_te, train_dataset, test_dataset = \
    U.prepare_data(x_train[:,:,None], y_train, x_test[:,:,None], y_test, method='hmc')

prior_params = {
    'prior_sigma_1': np.float32(np.exp(0)), 
    'prior_sigma_2': np.float32(np.exp(-7)), 
    'prior_pi': 0.5 
}

bnn_hmc_inst = M.BNN_HMC(prior_params, output_size=2)
out = bnn_hmc_inst.train(x_tr, y_tr, num_samp=10, step_size=3e-4, num_l_steps=100)
print('HMC acceptance rate:', np.mean(out[1].is_accepted.numpy()))
print(out[1].is_accepted)

y_pred = np.array(bnn_hmc_inst.posterior_predictive(data, out.all_states))
y_pred = 1-np.mean(y_pred, axis=0)
plt.scatter(X,Y,c=y_pred)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='gray', s=3, alpha=0.2)
plt.savefig(save_dir+'map_hmc')
plt.close()






