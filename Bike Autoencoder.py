import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
from PIL import Image
import time

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear( #applies a linear transformation to incoming data
            in_features = kwargs["input_shape"], out_features = 64 
        )
        self.encoder_output_layer = nn.Linear(
            in_features = 64, out_features = 64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features = 64, out_features = 64
        )
        self.decoder_output_layer = nn.Linear(
            in_features = 64, out_features = kwargs["input_shape"]
        )

    def forward(self, features, given_code = True, code = None, return_code = False):
        if code is None:
            activation = self.encoder_hidden_layer(features)
            activation = torch.relu(activation) #applies rectified linear unit function element-wise
            code = self.encoder_output_layer(activation)
            code = torch.relu(code)
        if return_code:
            return code
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

# class conAE(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.encode_conv1 = nn.Conv2d(1, 10, 5)
#         self.encode_conv2 = nn.Conv2d(10, 20, 5)
#         self.encode_linear1 = 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()
#model from autoencouder class

model = AE(input_shape = 480000).to(device)

#create an optimizer object
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

#mean-squared error losss
criterion = nn.MSELoss()

#load data set
path = os.path.dirname(os.path.realpath(__file__))
path1 = path + '\image_folder'
path2 = path + '\image_folder 2'
transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels = 1), 
            torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(
    root = path1, transform=transform
)

test_dataset = torchvision.datasets.ImageFolder(
    root = path2, transform = transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size = 64, shuffle = True, num_workers = 4
)

if __name__ == '__main__':

    epochs = 2
    print("Training:")
    loss_data = []
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:

            batch_features = batch_features.view(64, -1).to(device)
         
            optimizer.zero_grad()

            outputs = model(batch_features)

            train_loss = criterion(outputs, batch_features)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()
            

        loss = loss/ len(train_loader)
        loss_data.append(loss)
        print("Epoch : {}/{}, Loss = {:.6f}".format(epoch +1, epochs, loss))

    plt.plot(list(range(1,epochs + 1)), loss_data, marker = 'o', color = 'b', linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over {} Epochs".format(epochs))
    plt.show()
    plt.savefig('200 epochs loss')
    plt.close()


    print("Testing:")
    loss = 0
    count = 0
    orig_num = []
    recon_num = []
    for data, target in test_loader:

        data = data.view(64, -1).to(device)

        test_output = model(data)
 
        test_loss = criterion(test_output, data)

        loss += train_loss.item()
        count += 1
        if count < 4:
            orig_num.append(data)
            recon_num.append(test_output)

    loss /= len(test_loader)
    print("Average Test Set Loss = {:.6f}".format(loss))

    print(len(orig_num))
    fig, axs = plt.subplots(nrows = 2,ncols = 2, figsize = (14,7))
    for i in range(2):
        start = orig_num[i]
        end = recon_num[i]
        start = start.view(-1, 1, 600, 800).to(device)
        end = end.view(-1, 1, 600, 800).to(device)
        _ = axs[0,i].imshow(start[0][0], cmap = 'gray', interpolation = 'none')
        _ = axs[1,i].imshow(end[0][0].detach().numpy(), cmap = 'gray', interpolation = 'none')
    fig.suptitle("Bike Autoencoder Output: Top = Original, Bottom = Reconstructed")
    plt.show()    
    plt.savefig('50 epochs images')

    print('Time Passed: ' + str(time.time() - t0))

##tSNE Graph

    # tsne_test = torch.utils.data.DataLoader(
    # test_dataset, batch_size = 500, shuffle = False, num_workers = 4
    # )
    # perplexity = 50
    # tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = 0)
   
    # data, target = next(iter(tsne_test))
    # data = data.view(-1 , 784).to(device)
    # x = data.numpy()
    # y = target.numpy()

    # digits = datasets.load_digits()
    # target_ids = range(len(digits.target_names))

    # X_2d = tsne.fit_transform(x)
    # plt.figure(figsize = (6,5))
    # colors = 'r', 'g', 'b', 'b', 'm', 'y', 'k', 'gray', 'orange', 'purple'
    # for i, c, label in zip(target_ids, colors, digits.target_names):
    #     plt.scatter(X_2d[y == i,0], X_2d[y == i, 1], c=c, label = label)
    # plt.legend()
    # plt.title("tSNE Graph of MNIST Number Set: Perplexity = {}".format(perplexity))
    # plt.show()
