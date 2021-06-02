from sklearn.datasets import make_circles, make_moons
import numpy as np

# %%
def generate_circles(n_samples=10000, noise=.06, factor=.6, testing=True):

    x_train, y_train = make_circles(n_samples=n_samples, shuffle=True, 
                        noise=noise, random_state=None, factor=factor)

    if testing:

        x_part, y_part, x_rest, y_rest = [], [], [], []
        
        for i in range(n_samples):
            if not (x_train[i,0] < 0 and x_train[i,1] < 0):
                x_part.append(x_train[i,:])
                y_part.append(y_train[i])
            else:
                x_rest.append(x_train[i,:])
                y_rest.append(y_train[i])
                
        x_train = np.array(x_part) # training is over three quadrants
        y_train = np.array(y_part)
        x_test = np.array(x_rest) # testing is on the left out third quadrant
        y_test = np.array(y_rest)

    if not testing:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = x_train[0:2,:]
        y_test = y_train[0:2]

    return x_train, y_train, x_test, y_test

def generate_moons(n_samples=10000, noise=.06):

    x_train, y_train = make_moons(n_samples=n_samples, noise=noise, shuffle=True, random_state=None)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = x_train[0:2,:]
    y_test = y_train[0:2]

    return x_train, y_train, x_test, y_test
