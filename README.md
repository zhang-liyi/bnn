# Bayesian neural networks
An implementation of Bayesian neural networks using Variational Inference, Monte Carlo Dropout, and Hamiltonian Monte Carlo, with training on two toy datasets.

To run, use:
`python main.py --dataset=[D]`
where D is either 'circles' or 'moons'. VI works on MNIST too, but I'm currently working on the rest of methods to enable running on MNIST. The program saves one figure for each method, each figure showing the predictive distribution of each point in the domain.
