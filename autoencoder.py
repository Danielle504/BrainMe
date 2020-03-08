import torch
from torch import optim
import torch.nn as neural_net
from nibabel import nifti1
import numpy
import os
torch.backends.cudnn.deterministic = True
# Autoencoder class inherits standard PyTorch neural net class
# and produces trainable encoder/decoder pair. Uses standard
# multiple linear regression networks for this task.
class Autoencoder(neural_net.Module):
    # Our end goal is to have a pair of neural networks that 
    # compare encoded->decoded data with the original data. 
    # Thus, the input data by itself is our full training dataset.
    def __init__(self, dataset_folder, data_length, output_size):
        # data_length is the length of the data array of every tensor (a square tensor is assumed).
        super(Autoencoder, self).__init__()
        self.dataset_folder = dataset_folder
        self.data = []
        try:
            for file in os.listdir(dataset_folder): # Loads file locations, but not files, into memory.
                self.data.append(os.path.join(dataset_folder, file))
        except FileNotFoundError:
            print("\nDataset filepath is not found. Check working directory or provide absolute filepath as Python string literal.")
            return None
        if not isinstance(data_length, int):
            print("\n'Initial dimensions' is not an integer. Please specify only integer values or use the int() function.")
            return None
        if not isinstance(output_size, int):
            print("\n'Final dimensions' is not an integer. Please specify only integer values or use the int() function.")
            return None
        self.encoder = neural_net.Sequential(
            # neural_net.Linear() creates an active layer that changes an input 
            # from one length to another (this is referred to as 'dimensions').
            # neural_net.Linear(3,1) creates a layer that downsamples an input from
            # 3 values to 1, and neural_net.Linear(1,3) upsamples from 1 value to 3.
            # In our case, we are using 256x256x128 images, making for 256*256*128 inputs, 
            # which equals a first layer size of 8,388,608. We're going to reduce this
            # to an output of 128 over 6 layers.
            # Layer 1: 8388608 -> 4096.
             # Layer 2: 131072 -> a.
            neural_net.Linear(256*256*128, 1),
            neural_net.Sigmoid()
        )
        #for x in range(1, num_layers):
        #        neural_net.Linear(int(data_length/x), int(data_length/x+1)),
        #        neural_net.Sigmoid(),
        self.decoder = neural_net.Sequential(
            neural_net.Linear(1, 256*256*128),
            neural_net.Sigmoid()
            # Layer 6: 4096 -> 8338608.
        )

    def loss_function(self):
        return neural_net.MSELoss() # Initialise mean squared error loss class.

    def train_autoencoder(self, network_parameters, learning_rate):
        # network_parameters = Autoencoder(folder, 10, 2).parameters()
        optimiser = optim.SGD(network_parameters, lr=learning_rate, momentum=0.5)
        loss_function = self.loss_function() # Self function is used to allow for effective class override
        iterator = 0
        dataset_size = str(len(self.data))
        for data in self.data:
            if os.path.splitext(data)[1] == 'hdr':
                continue
            data = nifti1.load(data)
            data = numpy.array(data.dataobj)
            data = data.astype(numpy.float32)
            original_data = torch.from_numpy(data)
            original_data = original_data.reshape((8388608))
            encoded_data = self.encoder(original_data)
            encoded_decoded_data = self.decoder(encoded_data)
            loss = loss_function(encoded_decoded_data, original_data)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            iterator += 1
            print('Training iteration: [' + str(iterator) + ']/[' + dataset_size + ']. Current error: ' + str(loss.data.item()))

    def test_autoencoder(self):
        return 0

    def save_trained_model(self, filepath, save_encoder=True, save_decoder=True):
        if save_encoder:
            torch.save(self.encoder.state_dict(), filepath + '/encoder.pth')
        if save_decoder:
            torch.save(self.decoder.state_dict(), filepath + '/decoder.pth')
            
auc = Autoencoder('../input/oasis2', 8388608, 128)
auc.train_autoencoder(auc.parameters(), 0.01)
