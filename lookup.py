import numpy as np
from cv2 import cv2
from PIL import Image
from torch.utils import data
from time import sleep 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import copy
import random
from PIL import Image


def get_annotations(annotations):
    patterns = []
    pattern_dimensions = [0, 0]     # 0- width, 1- height
    antipatterns = []
    antipattern_dimensions = []     # 0- width, 1- height

    for annotation in annotations:
        img = np.array(Image.open(annotation['document']).convert('L'))
        subregion = img[int(annotation['topY']):int(annotation['bottomY']), int(annotation['topX']):int(annotation['bottomX'])]
        if annotation['is_antipattern'] == True:
            antipatterns.append(subregion)
            antipattern_dimensions.append([subregion.shape[1], subregion.shape[0]])
        
        else:
            patterns.append(subregion)
            pattern_dimensions[0] += subregion.shape[1]
            pattern_dimensions[1] += subregion.shape[0]
    
    pattern_dimensions[0] = int(pattern_dimensions[0]/len(patterns))
    pattern_dimensions[1] = int(pattern_dimensions[1]/len(patterns))
    
    for i in range(len(patterns)):
        patterns[i] = cv2.resize(patterns[i], tuple(pattern_dimensions), interpolation=cv2.INTER_AREA)

    patterns = np.array(patterns)
    antipatterns = np.array(antipatterns)

    return (patterns, pattern_dimensions, antipatterns, antipattern_dimensions)

def sp_noise(image, prob):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i][j] = 255
    return image

def generate_pattern_canvas(canvas_height, canvas_width, p_dataset, p_dimension, probability):
    canvas = np.zeros((canvas_height, canvas_width))
    canvas = sp_noise(canvas, 0.05)
    labels = np.zeros((canvas_height, canvas_width))
    
    i = 5
    while i < canvas_height:
        j = 5
        while j < canvas_width:
            prob = random.uniform(0,1)
            if(prob <= probability):
                number_of_images = len(p_dataset)
                random_image_number = random.randint(0, number_of_images - 1)
                sub_image = p_dataset[random_image_number]
                (_, sub_image) = cv2.threshold(sub_image, 127, 255, cv2.THRESH_BINARY)

                if(i+p_dimension[1] < canvas_height and j+p_dimension[0] < canvas_width):
                    canvas[i:i+p_dimension[1], j:j+p_dimension[0]] = sub_image
                    labels[i+(p_dimension[1]//2), j+(p_dimension[0]//2)] = 199920

            j += p_dimension[1] + 5
        i += p_dimension[0] + 5
        
    canvas[canvas <= 127] = 0
    canvas[canvas > 127] = 255

    return (canvas, labels)

def generate_antipattern_canvas(canvas_height, canvas_width, ap_dataset, ap_dimensions, p_dimension, anc, apc):
    canvas = np.zeros((canvas_height, canvas_width))
    canvas = sp_noise(canvas, 0.05)
    labels = np.zeros((canvas_height, canvas_width))

    i = 5
    while i < canvas_height:
        max_height = 0
        j = 5
        while j < canvas_width:
            temp_height = 0
            temp_width = 0
            if anc == -1:
                sub_image = np.zeros((p_dimension[1], p_dimension[0]))
                temp_height = p_dimension[1]
                temp_width = p_dimension[0]
            elif len(ap_dataset) > 0:
                sub_image = ap_dataset[anc]
                temp_height = ap_dimensions[anc][1]
                temp_width = ap_dimensions[anc][0]
            else:
                sub_image = np.zeros((ap_dimensions[0][1], ap_dimensions[0][0]))
                temp_height = p_dimension[1]
                temp_width = p_dimension[0]

            (_, sub_image) = cv2.threshold(sub_image, 127, 255, cv2.THRESH_BINARY)
            if(i+temp_height < canvas_height and j+temp_width < canvas_width):
                canvas[i:i+temp_height, j:j+temp_width] = sub_image

            if temp_height > max_height:
                max_height = temp_height

            j += temp_width + 5
            anc += 1
            if anc == len(ap_dataset):
                anc = -1
                apc += 1
        
        i += max_height + 5

    return canvas, labels, anc, apc

def make_dataset(p_dataset, p_dimension, ap_dataset, ap_dimensions, no_of_images = 100, probability = 0.6):
    X = []
    y = []

    anc = -1
    apc = 0
    for i in range(no_of_images):
        if apc < 20:
            image, label, anc, apc = generate_antipattern_canvas(300, 300, ap_dataset, ap_dimensions, p_dimension, anc, apc)
        else:
            image,label =  generate_pattern_canvas(300, 300, p_dataset, p_dimension, probability)
        X.append([image])
        y.append(label.reshape(1, -1))

    X, y = torch.from_numpy(np.stack(X)), torch.from_numpy(np.vstack(y))
    return (X, y)

class cnn(nn.Module):
    def __init__(self, kernel_dimension):
        super(cnn,self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 10, (kernel_dimension[0], kernel_dimension[1]), padding = ((kernel_dimension[0]-1)//2, (kernel_dimension[1]-1)//2)),
            nn.ReLU(),
            nn.Conv2d(10, 1, 1, padding=(0,0)),
            nn.ReLU(),
        )

    def forward(self,x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        return x

def train_model(train_loader, kernel_dimension = [30, 30], max_epochs = 1):
    first_epoch_loss = None
    last_epoch_loss  = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = cnn(kernel_dimension)

    if torch.cuda.device_count() > 1:
      net = nn.DataParallel(net)

    net.to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(net.parameters(), lr = 0.001)
    best_model = None
    min_loss = sys.maxsize

    for epoch in range(max_epochs):
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = net(X)
            loss = loss_fn(out, y)

            if(loss < min_loss):
                best_model = copy.deepcopy(net)
                min_loss = loss

            loss.backward()
            opt.step()

        print('Epoch: {: >2}/{: >2}  loss: {}'.format(epoch, max_epochs, loss))

        if(epoch == 0):
            first_epoch_loss = int(loss)
        elif(epoch == max_epochs - 1):
            last_epoch_loss = int(loss)
    
    return (best_model, min_loss, first_epoch_loss, last_epoch_loss)

def lookup(annotations, model_name):
    patterns, pattern_dimensions, antipatterns, antipattern_dimensions = get_annotations(annotations)
    patterns = 255 - patterns
    antipatterns = 255 - antipatterns

    (train_x, train_y) = make_dataset(p_dataset = patterns, p_dimension = pattern_dimensions, ap_dataset = antipatterns, ap_dimensions = antipattern_dimensions, no_of_images = 200)
    train_x = train_x.type(torch.float32)
    train_y = train_y.type(torch.float32)
    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(train_dataset, batch_size = 4, shuffle = True)
    
    kernel_width = pattern_dimensions[0]
    kernel_height = pattern_dimensions[1]

    if(kernel_height%2 == 0):
        kernel_height += 1

    if(kernel_width%2 == 0) :
        kernel_width += 1

    totalIterations = 2
    currentIteration = 0
    true_learning = False
    while currentIteration < totalIterations:
        print('Iteration =>', currentIteration)
        currentIteration += 1
        best_model, loss, first_loss, last_loss = train_model(train_loader, max_epochs = 20, kernel_dimension = [kernel_height, kernel_width])

        if((first_loss*0.3) >= last_loss):
            true_learning = True
            break
    
    print('Final Loss =',int(loss))

    if(true_learning):
        torch.save(best_model.state_dict(), 'trained_models/' + model_name + '.pth')
        return (True, [kernel_height, kernel_width])
    else:
        return (False, [0,0])