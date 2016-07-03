'''
Created on Nov 19, 2015

@author: ky
'''
import math
import random
import sys

random.seed(0)


def rand(a, b):
    """
    Create a random number between a and b
    """
    return (b - a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    """
    Create a matrix
    """
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def randomizeMatrix(matrix, a, b):
    """
    Random initialize matrix
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    """
    Sigmoid function prime
    """
    return y * (1 - y)


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        """
        Build NN
        """
        self.ni = ni + 1  # Add bias
        self.nh = nh + 1  # Add bias
        self.no = no

        # Output
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # Weight matrix
        self.wi = makeMatrix(self.ni, self.nh)  # from input to hidden layer
        self.wo = makeMatrix(self.nh, self.no)  # from hidden layer to output layer
        # Random initialize weights
        randomizeMatrix(self.wi, -0.2, 0.2)
        randomizeMatrix(self.wo, -2.0, 2.0)
        # Gradient matrix
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def runNN(self, inputs):
        """
        Feedforward
        """
        if len(inputs) != self.ni - 1:
            print 'incorrect number of inputs'

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh - 1):
            sumup = 0.0
            for i in range(self.ni):
                sumup += ( self.ai[i] * self.wi[i][j] )
            self.ah[j] = sigmoid(sumup)
        self.ah[self.nh - 1] = 1.0

        for k in range(self.no):
            sumup = 0.0
            for j in range(self.nh):
                sumup += ( self.ah[j] * self.wo[j][k] )
            self.ao[k] = sigmoid(sumup)

        return self.ao


    def backPropagate(self, targets, N, M):
        """
        Back-Propagation algorithm
        """
       

        # Output layer delta
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

        # Update output layer weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        # Hidden layer delta
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        # Update input layer weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change
        
        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        """
        Print weights
        """
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
        print ''

    def test(self, patterns):
        """
        Test function
        """
        test_results = 0
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            output = self.runNN(inputs)
            if output.index(max(output)) == targets.index(max(targets)):
                test_results += 1
        print test_results
        
    def train(self, patterns, max_iterations=200, N=0.5, M=0.1):
        """
        Training function
        """
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            if i % 50 == 0:
                print 'Combined error', error
        #self.test(patterns)

    def classification(self,keys,images):
        """
        Classification function
        """
        for i in range(len(images)):
            inputs = images[i][0]
            output = self.runNN(inputs)
            print  keys[i],'\t',output.index(max(output)) + 1

def import_trainingdata(filename,faces):
    """
    Import image data into a dictionary
    """
    with open(filename) as facefile:
        for line in facefile:
            if len(line) > 1 and not line.startswith('#'):
                if line.startswith('I'):
                    image_number = line.strip()
                    faces[image_number] = []
                else:
                    values = line.split()
                    values = [float(n)/32 for n in values]
                    faces[image_number].extend(values)
 
def import_facitdata(filename,facits):
    """
    Import facit data into a dictionary
    """
    with open(filename) as facit:   
        for line in facit:
            if len(line) > 1 and not line.startswith('#'):
                data = line.split()
                facits[data[0]] = [0]*4
                facits[data[0]][int(data[1]) - 1] = 1

def main():
    training_faces = {}
    training_facits = {}
    classification_faces = {}
    test_faces = {}
    test_results = {}
    training_set = []
    test_set = []
    classification_set = []
    filename = []
    
    for i in range(len(sys.argv)-1):
        filename.append(sys.argv[i+1])

    import_trainingdata(filename[0],training_faces)
    import_facitdata(filename[1],training_facits)
    import_trainingdata(filename[2],classification_faces)
    #import_facitdata(filename[3],test_results)
    
    keys = training_faces.keys()
    keys.sort()
    for item in keys:
        training_set.append([training_faces[item],training_facits[item]])
    ckeys = classification_faces.keys()
    ckeys.sort()
    for item in ckeys:
        #test_set.append([classification_faces[item],test_results[item]])
        classification_set.append([classification_faces[item]])
    
    myNN = NN(400, 20, 4)
    myNN.train(training_set)
    #myNN.test(test_set)
    myNN.classification(ckeys, classification_set)


if __name__ == "__main__":
    main()
