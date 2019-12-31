import numpy as np
from datetime import datetime



# activation function and their derivatives:
def ReLU(x):
    """
    relu activation function.
    :param x:
    :return: maximum of 0 and x.
    """
    return np.maximum(x, 0)

def ReLU_tag(x):
    return np.maximum(x, np.sign(x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - max(x))
    return e_x / e_x.sum()


# neural network class:
class NeuralNetwork:
    def __init__(self, neurons_per_layer):
        self.neurons_dimensions = neurons_per_layer
        self.num_layers = len(neurons_per_layer)

        # generates random b,w for each layer.

        #self.biases = [np.random.randn(d) for d in neurons_per_layer[1:]]
        self.biases = [np.zeros(d) for d in neurons_per_layer[1:]]
        #self.weights = [np.random.randn(d1, d0) for d0, d1 in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]
        self.weights = [np.sqrt(2/d0) * np.random.randn(d1, d0) for d0, d1 in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]

    def train(self, data_set, epochs, eta, validation_set=None):

        # define constants:
        alpha = 100 * (eta / len(data_set))
        n = len(data_set)


        # for num of epochs:
        for i in range(epochs):

            # moving between subgroups of the set:
            np.random.shuffle(data_set)
            for x, y in data_set:
                nabla_w, nabla_b = self.backprop(x, y)
                # nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                # nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [w - alpha * nw for w, nw in zip(self.weights,  nabla_w)]
                self.biases = [b - alpha * nb for b, nb in zip(self.biases, nabla_b)]

            # if validation_set:
                # print('epoch =', i, ': succsess rate:', self.validate(validation_set) * 100, '%')

    def forward(self, v):
        """
        forward propagation - pass x from 1st layer - input to the last - output, through hidden layers.
        :param x:  the input to the 1st layer.
        :return: vector of classification probabilites (after softmax).
        """
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, v) + b
            v = ReLU(z)
        return softmax(z)


    def backprop(self, x, y):

        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        # 1. feed forward  x.

        activations = [x.reshape(-1, 1)]  # list to store all the activations, layer by layer
        v = x
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, v) + b
            zs.append(z)
            v = ReLU(z)
            if len(list(v.shape)) == 1:
                activations.append(v.reshape(-1, 1))
            else:
                activations.append(v)
        p = softmax(z)

        # 2. compute backwards gradiants.
        E = p - y
        Grad = E

        for l in range(1, self.num_layers):
            d_w[-l] = np.dot(Grad.reshape(-1, 1), activations[-l - 1].T)
            d_b[-l] = Grad
            E = np.dot(self.weights[-l].T, Grad)

            if l < self.num_layers - 1:
                Grad = ReLU_tag(zs[-l - 1]) * E

        return (d_w, d_b)

    def validate(self, set):
        c = 0
        for x, y in set:
            p = self.forward(x)
            if np.argmax(p) == np.argmax(y): c += 1
            #print('prediction = {0}, real = {1}'.format(np.argmax(p), np.argmax(y)))
        return c/len(set)




# main:
def main():
    startTime = datetime.now()

    # open training x, y sets, test x set.
    with open('train_x') as train_x_file, open('train_y') as train_y_file:
        # x_lines = [train_x_file.readline() for _ in range(55000)]
        # y_lines = [train_y_file.readline() for _ in range(55000)]
        train_x_set = np.loadtxt(train_x_file) / 255
        train_y_set = [np.eye(1, 10, int(y) % 10)[0] for y in np.loadtxt(train_y_file)]
        data_set = [(x, y) for x, y in zip(train_x_set, train_y_set)]

    with open('test_x') as test_x_file:
        test_x_set = np.loadtxt(test_x_file) / 255

    # devide data_set to training and validation:
 #   np.random.shuffle(data_set)
    validation_percentage = 0.15
#    training_set = data_set[:int(validation_percentage * len(data_set))]
    validation_set = data_set[:int(validation_percentage * len(data_set))]

    input_dim = 28 * 28
    output_dim = 10

    network = NeuralNetwork([input_dim, 84, output_dim])
    network.train(data_set, epochs=20, eta=1, validation_set=validation_set)

    # write predictions to test y file:
    with open('test_y', 'w+') as test_y_file:
        for x in test_x_set:
            print(int(np.argmax(network.forward(x))), file=test_y_file)

    duration_secods = datetime.now() - startTime
    # print('run took', duration_secods, '(hours: minutes: seconds)')

if __name__ == "__main__":
    main()
