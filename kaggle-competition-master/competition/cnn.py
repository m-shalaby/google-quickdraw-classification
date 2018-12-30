import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

params = {
    'conv_layer_size': [50, 25],
    'conv_kernel_size': [5,3],
    'full_layer_size': [512,256,31],    
    'dropout_p': 0.25,
    'pool_size': 2,
    'image_size': 100
}

class ConvNet():
    '''
    To do: 
        1) DONE: finish the create_cnn function
        2) DONE: finish the image_size_full_layer function
        3) DONE: create the training algorithm
        4) create the testing algorithm
                NEED TO CHANGE THE MODEL_OUTPUT TO THE SAME FORMAT AS THE TEST_LABELS
        5) DONE: evaluation metrics
        6) test out different params?
    '''
    def __init__(self, params):

        self.conv_layer_size = params[0]
        self.conv_kernel_size = params[1]
        self.full_layer_size = params[2]
        self.dropout_p = params[3]
        self.pool_size = params[4]
        self.image_size = params[5]
        self.view = View()
        
    def create_cnn(self):
        '''
        conv_tuples is a list of tuples, each tuple is of 3 dimensions representing one conv layer:
            (input_size, output_size, filter_size)
        Set input to first layer to be of 3 dimensions.
        
        #Need to do: 
                1) combine the full network
        '''
        
        #Creating the convolutional layers

        self.pool = nn.MaxPool2d(self.pool_size)

        self.dropout = nn.Dropout2d(p=self.dropout_p)

        self.conv_layers=[]
        for i in range(len(self.conv_layer_size)):
            if i > 0:
                self.conv_layer_tuples = self.conv_layer_tuples+[(self.conv_layer_size[i-1], self.conv_layer_size[i], self.conv_kernel_size[i])]
            else:
                self.conv_layer_tuples = [(1, self.conv_layer_size[i], self.conv_kernel_size[i])]
            self.conv_layers = self.conv_layers+[nn.Conv2d(self.conv_layer_tuples[i][0], self.conv_layer_tuples[i][1], self.conv_layer_tuples[i][2])]
        # print(self.conv_layers)

        self.image_size_full_layer()

        #Creating the full layers

        full_layer_input_dim = int(self.conv_layer_size[-1] * self.image_size**2)

        self.full_layers=[]
        for i in range(len(self.full_layer_size)):
            if i > 0:
                self.full_layer_tuples = self.full_layer_tuples+[(self.full_layer_size[i-1], self.full_layer_size[i])]
            else:
                self.full_layer_tuples = [(full_layer_input_dim, self.full_layer_size[i])]
            self.full_layers = self.full_layers+[nn.Linear(self.full_layer_tuples[i][0], self.full_layer_tuples[i][1])]
        # print(self.conv_layers)
        # print(self.full_layers)

        # Combine convoluted and hidden layers to create full network

        c_nn = []
        for layer in self.conv_layers:
            c_nn.append(layer)
            c_nn.append(nn.ReLU())
            c_nn.append(self.pool)

        c_nn.append(self.view)

        for layer in self.full_layers[:-1]:
            c_nn.append(layer)
            c_nn.append(nn.ReLU())
            c_nn.append(self.dropout)

        c_nn.append(self.full_layers[-1])

        # for i in c_nn:
            # print(i)

        return(c_nn)

    def image_size_full_layer(self):
        '''
        The purpose of this function is to compute the input size of the first full layer after the convolution layers.
        '''

        #output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        for layer in self.conv_layers:
            self.image_size = int(self.image_size - layer.kernel_size[0]) / layer.stride[0] + 1
            # print(self.pool.stride)
            self.image_size = int(self.image_size - self.pool.kernel_size) / self.pool.stride + 1
            # print(self.image_size)

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1) 

class Trainer():
    def __init__(self, model, criterion, learning_rate, num_epochs, batch_size, image_size):
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size

    def train(self, train_images, train_labels):
        '''
        This routine trains the ConvNet model based on the training data.
        '''
        training_data = torch.Tensor(train_images.reshape(-1, 1, self.image_size, self.image_size).astype(np.float))
        output = []
        for label in train_labels:
            vector = [0]*31
            vector[int(label)] = 1
            output = output + [vector]
        training_labels = torch.Tensor(output)
        dataset = TensorDataset(training_data, training_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        z=1
        for epoch in range(self.num_epochs):
            for images, labels in dataloader:
                images, labels = Variable(images), Variable(labels)

                print(z)
                z+=1

                print(images.shape, labels.shape)
                
                optimizer.zero_grad()
                model_output = self.model(images)              
                _, classes = torch.max(labels, 1)

                print(model_output.shape)
                print(labels.shape)
                print(classes)

                loss = self.criterion(model_output, classes)
                loss.backward()
                optimizer.step()

        print(self.model)
        return self.model


    def test(self, trained_model, test_images, test_labels):
        '''
        This routine generates the labels for the test data.
        '''
        testing_images = torch.Tensor(test_images.reshape(-1, 1, self.image_size, self.image_size).astype(np.float))
        testing_images = Variable(testing_images)
        model_output = trained_model(testing_images)
        model_output = model_output.data
        model_output = model_output.numpy()

        output = []
        for label in model_output:
            output.append(np.argmax(label))
        model_output = np.array(output, dtype='int')
        
        accuracy = accuracy_score(test_labels, model_output)
        precision = precision_score(test_labels, model_output, average='micro')
        recall = recall_score(test_labels, model_output, average='micro')
        f1 = f1_score(test_labels, model_output, average='micro')
        conf_matrix = confusion_matrix(test_labels, model_output)
        
        print(test_labels)
        print(model_output)

        return accuracy, precision, recall, f1, conf_matrix

convnet_class = ConvNet([params['conv_layer_size'], params['conv_kernel_size'], params['full_layer_size'], params['dropout_p'], params['pool_size'], params['image_size']])
convnet_layers = convnet_class.create_cnn()
convnet = nn.Sequential(*convnet_layers)
# print(convnet)

train_images = data.load_image_array('train_images.npy')[1000:]
train_labels = data.load_categories('train_labels.csv')[1000:]

test_images = data.load_image_array('train_images.npy')[:1000]
test_labels = data.load_categories('train_labels.csv')[:1000]

trainer = Trainer(convnet, nn.CrossEntropyLoss(), 0.001, 10, 250, params['image_size'])
trained_convnet = trainer.train(train_images, train_labels)

a, p, r, f, c = trainer.test(trained_convnet, test_images, test_labels)
print(a, p, r, f, c)
