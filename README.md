# Image Perspective Transformation using Deep Learning

## Overview

This project focuses on performing perspective transformations on images using a deep learning model. The goal is to warp an input image to match a target perspective, which is particularly useful in applications like image stitching, augmented reality, and camera calibration. The project uses a custom ResNet-18 architecture to predict the transformation matrix required to warp the input image to the desired perspective.

## Features

- **Perspective Transformation**: The project uses a deep learning model to predict a 3x3 transformation matrix that can be applied to warp an input image to a target perspective.
- **Custom ResNet-18 Architecture**: A modified version of the ResNet-18 model is used to predict the transformation matrix.
- **Data Processing**: The project includes functions to process input images and annotations, extract key points, and compute the transformation matrix.
- **Training and Inference**: The model can be trained on a dataset of images and their corresponding transformation matrices, and it can be used to perform inference on new images.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.x**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **tqdm**

You can install the required Python libraries using pip:

```bash
pip install torch torchvision opencv-python numpy tqdm
```

## Project Structure

The project consists of the following components:

- **Data Processing**: Functions to load and process images and annotations, extract key points, and compute transformation matrices.
- **Model Architecture**: A custom ResNet-18 model designed to predict a 3x3 transformation matrix.
- **Training**: Code to train the model on a dataset of images and their corresponding transformation matrices.
- **Inference**: Code to perform inference on new images using the trained model.

## Usage

### Data Processing

The project includes functions to process input images and annotations. The `get_sample()` function loads an image and its corresponding annotation file, extracts key points, and computes the transformation matrix required to warp the image to the target perspective.

```python
def get_sample():
    # Load a random sample from the dataset
    indx = np.random.randint(0, len(data_set))
    target_file = data_set[indx][:-4]
    with open('Annos/{}.txt'.format(target_file), 'r') as file:
        file_content = file.read()
    
    # Extract key points and compute transformation matrix
    trap_points = {}
    trap_points['top_left'] = get_pair(file_content.split('\n')[1])
    trap_points['top_right'] = get_pair(file_content.split('\n')[2])
    trap_points['bottom_right'] = get_pair(file_content.split('\n')[3])
    trap_points['bottom_left'] = get_pair(file_content.split('\n')[4])
    center_x = (trap_points['top_left'][0] + trap_points['top_right'][0] + trap_points['bottom_right'][0] + trap_points['bottom_left'][0])/4
    center_y = (trap_points['top_left'][1] + trap_points['top_right'][1] + trap_points['bottom_right'][1] + trap_points['bottom_left'][1])/4
    trap_points['center'] = [center_x, center_y]
    
    rect_points = {}
    rect_points['top_left'] = get_pair(file_content.split('\n')[6])
    rect_points['top_right'] = get_pair(file_content.split('\n')[8])
    rect_points['bottom_right'] = get_pair(file_content.split('\n')[7])
    rect_points['bottom_left'] = get_pair(file_content.split('\n')[9])
    center_x = (rect_points['top_left'][0] + rect_points['top_right'][0] + rect_points['bottom_right'][0] + rect_points['bottom_left'][0])/4
    center_y = (rect_points['top_left'][1] + rect_points['top_right'][1] + rect_points['bottom_right'][1] + rect_points['bottom_left'][1])/4
    rect_points['center'] = [center_x, center_y]
    
    rotations = {}
    rotations['trap'] = float(file_content.split('\n')[11])
    rotations['rect'] = float(file_content.split('\n')[13])
    
    # Rotate points
    trap_points['top_left'] = rotate_point(trap_points['top_left'], trap_points['center'], rotations['trap'])
    trap_points['top_right'] = rotate_point(trap_points['top_right'], trap_points['center'], rotations['trap'])
    trap_points['bottom_right'] = rotate_point(trap_points['bottom_right'], trap_points['center'], rotations['trap'])
    trap_points['bottom_left'] = rotate_point(trap_points['bottom_left'], trap_points['center'], rotations['trap'])

    rect_points['top_left'] = rotate_point(rect_points['top_left'], rect_points['center'], rotations['rect'])
    rect_points['top_right'] = rotate_point(rect_points['top_right'], rect_points['center'], rotations['rect'])
    rect_points['bottom_right'] = rotate_point(rect_points['bottom_right'], rect_points['center'], rotations['rect'])
    rect_points['bottom_left'] = rotate_point(rect_points['bottom_left'], rect_points['center'], rotations['rect'])
    
    # Compute transformation matrix
    src_points = np.array([trap_points['top_left'], trap_points['top_right'], trap_points['bottom_right'], trap_points['bottom_left']], dtype=np.float32)
    dst_points = np.array([rect_points['top_left'], rect_points['top_right'], rect_points['bottom_right'], rect_points['bottom_left']], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    y = M.reshape(9)
    X = cv2.imread("GT/{}png".format(target_file[:-3]), 0)
    X[X > 10] = 255
    X[X <= 10] = 0
    X = X / 255.0
    
    # Normalize transformation matrix
    y[0] = y[0] * 10
    y[1] = y[1] * 10
    y[3] = y[3] * 10
    y[4] = y[4] * 10
    y[2] = y[2] / X.shape[0]
    y[5] = y[5] / X.shape[1]
    X = np.expand_dims(X, axis=0)
    X = np.expand_dims(X, axis=0)
    
    return X, y, "imgs/{}jpg".format(target_file[:-3])
```

### Model Architecture

The project uses a custom ResNet-18 model to predict the transformation matrix. The model is defined in the `ResNet_18` class, which includes a series of convolutional layers followed by fully connected layers to output the 3x3 transformation matrix.

```python
class ResNet_18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
```

### Training

The model can be trained using the following code:

```python
net = ResNet_18(1, 9)
net.cuda()

criteria = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for i in range(num_batches):
        X, y, _ = get_sample()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()
        
        optimizer.zero_grad()
        outputs = net(X)
        loss = criteria(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

### Inference

To perform inference on a new image, use the following code:

```python
with torch.no_grad():
    X, y, img_file = get_sample()
    X = torch.tensor(X, dtype=torch.float32).cuda()
    outputs = net(X)
    
    M_ = outputs.detach().cpu().numpy()[0]
    M_[2] = M_[2] * X.shape[0]
    M_[5] = M_[5] * X.shape[1]
    M_[0] = M_[0] / 10
    M_[1] = M_[1] / 10
    M_[3] = M_[3] / 10
    M_[4] = M_[4] / 10
    
    M_ = M_.reshape((3, 3))
    
    img = cv2.imread(img_file)
    warped_image = cv2.warpPerspective(img, M_, (GT_img.shape[1], GT_img.shape[0]))
    cv2.imwrite('output.jpg', 0.5 * warped_image + 0.5 * GT_img)
```

## Conclusion

This project provides a comprehensive solution for performing perspective transformations on images using a deep learning model. The custom ResNet-18 architecture is designed to predict the transformation matrix required to warp an input image to a target perspective. The project includes functions for data processing, model training, and inference, making it a complete solution for image perspective transformation tasks.
