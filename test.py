from math import sqrt

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

import minisom
from minisom import MiniSom


class TestMinisom:

    def setUp(self):
        self.som = MiniSom(5, 5, 1)
        for w in self.som.weights:  # checking weights normalization
            assert_almost_equal(1.0, np.linalg.norm(w))
        self.som.weights = np.zeros((5, 5))  # fake weights
        self.som.weights[2, 3] = 5.0
        self.som.weights[1, 1] = 2.0

    def test_fast_norm(self):
        assert minisom.fast_norm(np.array([1, 3])) == sqrt(1 + 9)

    def test_gaussian(self):
        bell = minisom.gaussian((2, 2), 1, self.som.neigx, self.som.neigy)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_win_map(self):
        winners = self.som.win_map([5.0, 2.0])
        assert winners[(2, 3)][0] == 5.0
        assert winners[(1, 1)][0] == 2.0

    def test_activation_reponse(self):
        response = self.som.activation_response([5.0, 2.0])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_quantization_error(self):
        self.som.quantization_error([5, 2]) == 0.0
        self.som.quantization_error([4, 1]) == 0.5

    def test_quantization(self):
        q = self.som.quantization(np.array([4, 2]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    # def test_random_seed(self):
    #     som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
    #     som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
    #     # same initialization
    #     assert_array_almost_equal(som1.weights, som2.weights)
    #     data = np.random.rand(100, 2)
    #     som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
    #     som1.train_random(data, 10)
    #     som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
    #     som2.train_random(data, 10)
    #     # same state after training
    #     assert_array_almost_equal(som1.weights, som2.weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = np.array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10)
        assert q1 > som.quantization_error(data)

    # def test_train_random(self):
    #     som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
    #     data = np.array([[4, 2], [3, 1]])
    #     q1 = som.quantization_error(data)
    #     som.train_random(data, 10)
    #     assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, sigma=0.1, random_seed=1)
        som.random_weights_init(np.array([[1.0, .0]]))
        for w in som.weights:
            assert_array_equal(w[0], np.array([1.0, .0]))
