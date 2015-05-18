import math
import numpy as np

from collections import defaultdict
from warnings import warn


"""
    Minimalistic implementation of the Self Organizing Maps (SOM).

    Giuseppe Vettigli 2013.
"""


# distances
def fast_norm(x, y=None):
    """
        Returns norm-2 of a 1-D numpy array.
        * faster than np.linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    if not y:
        y = x.T
    return math.sqrt(np.dot(x, y))

def cosine(x, y):
    """ Computes the Cosine distance between 1-D arrays. """
    return 1 - (np.dot(x, y) / (fast_norm(x) * fast_norm(y)))


# neighboorhood
def gaussian(c, sigma, neigx, neigy):
    """ Returns a Gaussian centered in c """
    d = 2 * np.pi * sigma * sigma
    ax = np.exp(-np.power(neigx - c[0], 2) / d)
    ay = np.exp(-np.power(neigy - c[1], 2) / d)
    return np.outer(ax, ay)  # the external product gives a matrix

def diff_gaussian(c, sigma, neigx, neigy):
    """ Mexican hat centered in c (unused) """
    xx, yy = np.meshgrid(neigx, neigy)
    p = np.power(xx - c[0], 2) + np.power(yy - c[1], 2)
    d = 2 * np.pi * sigma * sigma
    return np.exp(-(p) / d) * (1 - 2 / d * p)


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 neighborhood_fn=gaussian, random_seed=None):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            random_seed, random seed to use.
        """
        if sigma >= x / 2.0 or sigma >= y / 2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.random_generator = np.random.RandomState(random_seed)

        # weights random initialization and normalization
        self.weights = self.random_generator.rand(x, y, input_len) * 2 - 1
        self.weights = np.array([v / np.linalg.norm(v) for v in self.weights])

        self.activation_map = np.zeros((x, y))
        # neighborhood variables
        self.neigx = np.arange(x)
        self.neigy = np.arange(y)
        self.neighborhood = neighborhood_fn

    def _init_T(self, num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        # keeps the learning rate nearly constant for the first half of the iterations
        self.T = num_iteration / 2

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = np.subtract(x, self.weights)  # x - w
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def _winner(self, x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def _winner2(self, x):
        activation_map = np.apply_along_axis(lambda xx: cosine(x, xx), 2, self.weights)
        return np.unravel_index(activation_map.argmin(), activation_map.shape)

    def _update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        # eta(t) = eta(0) / (1 + t/T)
        # keeps the learning rate nearly constant for the first T iterations
        # and then adjusts it
        eta = self.learning_rate / (1 + t / self.T)
        # sigma and learning rate decrease with the same rule
        sig = self.sigma / (1 + t / self.T)
        g = self.neighborhood(win, sig, self.neigx, self.neigy) * eta  # improves the performances
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index] * (x - self.weights[it.multi_index])
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_iteration):
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            # pick a random sample
            rand_i = int(round(self.random_generator.rand() * len(data) - 1))
            self._update(data[rand_i], self._winner(data[rand_i]), iteration)

    def train_batch(self, data, epochs):
        """ Trains using all the vectors in data sequentially """
        self._epoch_weights = [] # store weights for each epoch
        num_iteration = len(data) * epochs
        # self._init_T(num_iteration)
        self.T = len(data) # keeps the learning rate nearly constant for the first epoch
        for i in range(epochs):
            for idx in xrange(len(data)):
                iteration = i+1 * idx+1
                self._update(data[idx], self._winner(data[idx]), iteration)
            self._epoch_weights.append(np.copy(self.weights))

    def distance_map(self):
        """
            Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1)
        """
        um = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += fast_norm(self.weights[ii, jj, :] - self.weights[it.multi_index])
            it.iternext()
        um = um / um.max()
        return um

    # unused
    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[int(self.random_generator.rand() * len(data) - 1)]
            self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

    # unused
    def activate(self, x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def quantization(self, data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = np.zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self._winner(x)]
        return q

    def quantization_error(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        for x in data:
            error += fast_norm(x - self.weights[self._winner(x)])
        return error / len(data)

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self._winner(x)] += 1
        return a

    def win_map(self, data):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self._winner(x)].append(x)
        return winmap


class MiniSomCosine(MiniSom):
    """ This is a specialised minisom class using a cosine distance function. """
    def __calculate_data_norm2(self):
        # since cosine distance makes use of norm2 values a lot, pre calculate them
        pass

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = cosine(x, self.weights[it.multi_index])
            it.iternext()

    def _update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        # eta(t) = eta(0) / (1 + t/T)
        # keeps the learning rate nearly constant for the first T iterations
        # and then adjusts it
        eta = self.learning_rate / (1 + t / self.T)
        # sigma and learning rate decrease with the same rule
        sig = self.sigma / (1 + t / self.T)
        g = self.neighborhood(win, sig, self.neigx, self.neigy) * eta  # improves the performances
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index] * (x - self.weights[it.multi_index])
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

    def distance_map(self):
        """
            Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1)
        """
        um = np.zeros(self.weights.shape[:2])
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += self._cosine(self.weights[ii, jj, :], self.weights[it.multi_index])
            it.iternext()
        um = um / um.max()
        return um

    def quantization_error(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        for x in data:
            y = self.weights[self.win_map(x)]
            error += self._cosine(x, y)
        return error / len(data)
