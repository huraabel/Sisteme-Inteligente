

import numpy as np
import unittest


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


class PerceptronTest(unittest.TestCase):

    def test_mimics_logical_and(self):
        weights = np.array([-1, 1, 1])

        a = 1
        b = 1
        inputs = np.array([a, b])

        perceptron = Perceptron(inputs.size)
        perceptron.weights = weights

        output = perceptron.predict(inputs)
        self.assertEqual(output, a & b)

    def test_trains_for_logical_and(self):
        labels = np.array([1, 0, 0, 0])
        input_matrix = []
        input_matrix.append(np.array([1, 1]))
        input_matrix.append(np.array([1, 0]))
        input_matrix.append(np.array([0, 1]))
        input_matrix.append(np.array([0, 0]))

        perceptron = Perceptron(2, threshold=10, learning_rate=1)
        perceptron.train(input_matrix, labels)

        a = 1
        b = 1
        inputs = np.array([a, b])

        output = perceptron.predict(inputs)
        self.assertEqual(output, a & b)

if __name__ == '__main__':
    unittest.main()