import sys
import os
import numpy as np
print('yes')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import time
import random
from scipy import stats

# torch.autograd.set_detect_anomaly(True)

class VAE(nn.Module):
    def __init__(self, latent_variable_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size = 3, padding = 1)
      
        self.fc1 = nn.Linear(4*256*256, 64)
        self.fc2m = nn.Linear(64, latent_variable_dim)
        self.fc2s = nn.Linear(64, latent_variable_dim)

        self.fc3 = nn.Linear(latent_variable_dim, 64)
        self.fc4 = nn.Linear(64, 4 * 256 * 256)
        self.conv3 = nn.ConvTranspose2d(4, 2, kernel_size = 3, padding = 1)
        self.conv4 = nn.ConvTranspose2d(2, 1, kernel_size = 3, padding = 1)
   
    def reparameterize(self, log_var, mu):
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s)
        return eps.mul(s).add_(mu)  

    def forward(self, input):
        x = input

        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))

        x = x.view(input.size()[0], -1)

        x = torch.relu(self.fc1(x))
        
        log_s = self.fc2s(x)
        m = self.fc2m(x)
        z = self.reparameterize(log_s, m)

        x = self.decode(z)

        return x, m, log_s, z

    def decode(self, z):
        x = torch.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))

        x = x.view(-1, 4, 256, 256)
        x = torch.relu(self.conv3(x))
        #print(x.shape)
        x = torch.relu(self.conv4(x))
        #print(x.shape)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()
#model from autoencouder class

model = VAE(40).to(device)

#create an optimizer object
optimizer = optim.Adam(model.parameters(), lr = 0.0000075)

#mean-squared error losss
criterion = nn.MSELoss()

#load data set
path = os.path.dirname(os.path.realpath(__file__))

path1 = path + '\image_folder 128'
path2 = path + '\image_folder 128'


transform = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels = 1),
            torchvision.transforms.Pad((0, 200, 0, 0)),
            torchvision.transforms.Resize((256,256)),
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

#####################################################################################
def no_background(batch):
    batch = np.array(batch)
    batch_list = []
    for image in batch:
        image.reshape(1, 256, 256)
        row_count = 0
        for row in image[0]:
            background_val = stats.mode(row)[0][0]
            row = np.where(row == background_val, 255, row)
            row = np.where(255> abs(row - background_val).any() > 170, 170, row )
            row = np.where(row != 255, 0, row)
            image[0][row_count,:] = row
            row_count += 1
        #image.reshape(1, 256, 256)
        batch_list.append(image)
        #print(image.shape)
    #plt.imshow(image, cmap = 'gray')
    #plt.show()
    new_batch = np.stack(batch_list, axis = 0)
    new_batch = torch.tensor(new_batch).float()
    return new_batch

def gauss(batch):
    batch = np.array(batch)
    batch_list = []
    for image in batch:
        image.reshape(1, 256, 256)
        ch, row, col = image.shape
        mean = 0
        gauss = 1 * np.random.normal(mean, 1, (ch, row, col))
        gauss = gauss.reshape(ch, row, col)
        #print(gauss.shape)
        noisy = image + gauss
        plt.imshow(noisy[0], cmap = 'gray')
        plt.show()
        batch_list.append(noisy)
    new_batch = np.stack(batch_list, axis = 0)
    new_batch = torch.tensor(new_batch).float()
    return new_batch


def train(epochs, plot = False):
    print("Training:")
    loss_data = []

    for epoch in range(epochs):
        loss = 0
        count = 0
        for batch_features, _ in train_loader:

            #print(type(batch_features))
            #batch1 = no_background(batch_features)

            new_batch = gauss(batch_features)
            #new_batch = noise3(batch_features, 0.15)

            #print(type(new_batch))

            optimizer.zero_grad()

            outputs, _, _, train_code = model(new_batch)


            train_loss = criterion(outputs, batch_features)

            train_loss.backward()

            optimizer.step()

            count += 1
            if np.isnan(train_loss.item()):
                print('NAN detected at Batch {}'.format(count))
            
            loss += train_loss.item()           

            ##print(loss)    

        loss = loss/ len(train_loader)
        loss_data.append(loss)
        print("Epoch : {}/{}, Loss = {:.6f}".format(epoch + 1, epochs, loss))

    if plot is True:
        plt.plot(list(range(1,epochs + 1)), loss_data, marker = 'o', color = 'b', linewidth=2.0)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over {} Epochs".format(epochs))
        plt.show()


def test(plot = False):
    print("Testing:")
    loss = 0
    epochs = 20
    orig_num = []
    recon_num = []
    test_codes = []
    for data, target in test_loader:

        no_back_data = no_background(data)

        new_data = gauss(no_back_data)

        test_output, _, _, test_code = model(new_data)
 
        test_loss = criterion(test_output, data)

        loss += test_loss.item()
        

        test_codes.append(test_code)
        orig_num.append(data)
        recon_num.append(test_output)

    loss /= len(test_loader)
    print("Average Test Set Loss = {:.6f}".format(loss))

    if plot is True:
        fig, axs = plt.subplots(nrows = 2,ncols = 2, figsize = (14,7))
        for i in range(2):
            start = orig_num[i]
            end = recon_num[i]
            start = start.view(-1, 1, 256, 256).to(device)
            end = end.view(-1, 1, 256, 256).to(device)
            start_im = start[0][0]
            end_im = end[0][0].detach().numpy()
            start_im = start_im[64:256, 0:256]
            end_im = end_im[64:256, 0:256]
            _ = axs[0,i].imshow(start_im, cmap = 'gray', interpolation = 'none')
            _ = axs[1,i].imshow(end_im, cmap = 'gray', interpolation = 'none')
        fig.suptitle("Bike Autoencoder Output ({} Epochs): Top = Original, Bottom = Reconstructed".format(epochs))
        plt.show()    

    return test_codes, orig_num, recon_num
    print('Time Passed: ' + str(time.time() - t0))


def nearestneigh(test_codes, orig_num, recon_num, plot = False):
    all_orig = torch.stack(orig_num)
    all_orig = all_orig.view(-1, 256*256)


    query_code = test_codes[0][0]
    query_code = query_code.view(1, -1)

    test_codes = torch.stack(test_codes)
    test_codes = test_codes.view(-1 ,40)

    test_codes = np.array(test_codes.detach().numpy())
    query_code = np.array(query_code.detach().numpy())

    nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree').fit(test_codes)

    distances, indices = nbrs.kneighbors(np.array(query_code))

    if plot is True:
        fig, axs = plt.subplots(nrows = 1, ncols = 5, figsize = (14, 4))
        for i in range(len(indices[0])):
            index = indices[0][i]
            end = all_orig[index]
            end = end.view(-1, 1, 256, 256).to(device)
            image = end[0][0].detach().numpy()
            image = image[64:256, 0:256]
            _ = axs[i].imshow(end[0][0].detach().numpy(), cmap = 'gray', interpolation = 'none')
        fig.suptitle("Autoencoder BikeCAD Dataset Nearest Neighbors")
        plt.show()    
    print('NearestNeighbors Completed')
    return test_codes, recon_num, indices


#tSNE Graph
def tsnegraph(test_codes, recon_num, orig_num):
    perplexity = 50
    tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = 0)

    test_codes_list = np.array(test_codes)


    all_orig = torch.stack(orig_num)
    all_orig = all_orig.view(-1, 256, 256)
    
    x = np.array(test_codes)

    bikes = '1', '2', '3'
    target_ids = range(2)
    colors = 'r', 'g', 'b'
    indexes = []
    index0 = []
    index1 = []
    index2 = []

    #clustering
    kmeans = KMeans(n_clusters = 3, random_state = 0).fit(x)
    y = kmeans.labels_


    count0 = 0
    count1 = 0
    count2 = 0
    total_count = 0

    while total_count < 12:
        n = random.randint(0, len(all_orig) - 1)
        value = y[n]
        if value == 0 and count0 < 12:
            if n in index0:
                continue   
            else:
                indexes.append(n)
                index0.append(n)
                count0 += 1
                total_count += 1
        if value == 1 and count1 < 4:
            if n in index1:
                continue   
            else:
                indexes.append(n)
                index1.append(n)
                count1 += 1
                total_count += 1
        if value == 2 and count2 < 4:
            if n in index2:
                continue   
            else:
                indexes.append(n)
                index2.append(n)
                count2 += 1
                total_count += 1



    fig, ax = plt.subplots()
    X_2d = tsne.fit_transform(x)

    print('About to plot: t-SNE Graph')

    for i, c, label in zip([0,1,2], colors, bikes):
        ax.scatter(X_2d[y == i,0], X_2d[y==i ,1], c = c, label = label)


    images0 = {}
    images1 = {}
    images2 = {}
    
    count = 0
    for x0, y0 in zip(X_2d[:,0], X_2d[:,1]):
        count += 1
        if count in indexes:
            image = all_orig[count]
            image = np.array(image)
            image = image[64:256, 0:256]
            
            ab = AnnotationBbox(OffsetImage(image, cmap = 'gray', zoom = 0.2), (x0, y0), frameon = False)
            ax.add_artist(ab)
            if count in index0:
                images0[x0] = image
            elif count in index1:
                images1[x0] = image
            else:
                images2[x0] = image

    plt.title('t-SNE Plot for BikeCAD Images')
    plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
    plt.show()
    print(indexes)
    print(images1.keys())

    x0_list0 = sorted(images0.keys())
    x0_list1 = sorted(images1.keys())
    x0_list2 = sorted(images2.keys())

    print(len(x0_list0))
    print(len(x0_list1))
    print(len(x0_list2))

    print(len(indexes))

    

    fig, axes = plt.subplots(nrows = 3, ncols = 4)
    for i in range(4):
        _ = axes[0,i].imshow(images0[x0_list0[i]], cmap = 'gray')
        _ = axes[1,i].imshow(images1[x0_list1[i]], cmap = 'gray')
        _ = axes[2,i].imshow(images2[x0_list2[i]], cmap = 'gray')

        if i == 3:
            continue
        else:
            axes[i,0].set_ylabel('Cluster {}'.format(str(i+1)), rotation = 0, size = 'large')
    plt.tight_layout()
    fig.suptitle('Bike Progression Through Clusters')
    plt.show()




if __name__ == '__main__':
    train(10)
    test_codes_list, orig_num, recon_num = test(plot = True)
    test_codes2, recon_num2, indices = nearestneigh(test_codes_list, orig_num, recon_num)
    tsnegraph(test_codes2, recon_num2, orig_num)