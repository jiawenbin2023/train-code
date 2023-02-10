# net structure
net structure  
CNN
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
- 
            Conv2d-1          [-1, 6, 222, 222]             168  
              ReLU-2          [-1, 6, 222, 222]               0  
         MaxPool2d-3          [-1, 6, 111, 111]               0  
            Conv2d-4         [-1, 16, 109, 109]             880  
              ReLU-5         [-1, 16, 109, 109]               0  
         MaxPool2d-6           [-1, 16, 54, 54]               0  
            Linear-7                  [-1, 120]       5,598,840  
              ReLU-8                  [-1, 120]               0  
            Linear-9                   [-1, 84]          10,164  
             ReLU-10                   [-1, 84]               0  
           Linear-11                   [-1, 20]           1,700  
             ReLU-12                   [-1, 20]               0  
           Linear-13                    [-1, 2]              42  
 ----------------------------------------------------------------
AlexNet
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
- 
            Conv2d-1           [-1, 64, 55, 55]          23,296  
              ReLU-2           [-1, 64, 55, 55]               0  
         MaxPool2d-3           [-1, 64, 27, 27]               0  
            Conv2d-4          [-1, 192, 27, 27]         307,392  
              ReLU-5          [-1, 192, 27, 27]               0  
         MaxPool2d-6          [-1, 192, 13, 13]               0  
            Conv2d-7          [-1, 384, 13, 13]         663,936  
              ReLU-8          [-1, 384, 13, 13]               0  
            Conv2d-9          [-1, 256, 13, 13]         884,992  
             ReLU-10          [-1, 256, 13, 13]               0  
           Conv2d-11          [-1, 256, 13, 13]         590,080  
             ReLU-12          [-1, 256, 13, 13]               0  
        MaxPool2d-13            [-1, 256, 6, 6]               0  
          Dropout-14                 [-1, 9216]               0  
           Linear-15                 [-1, 4096]      37,752,832  
             ReLU-16                 [-1, 4096]               0  
          Dropout-17                 [-1, 4096]               0  
           Linear-18                 [-1, 4096]      16,781,312  
             ReLU-19                 [-1, 4096]               0  
           Linear-20                    [-1, 2]           8,194  
![image](https://user-images.githubusercontent.com/123912496/215446342-e43cc868-4aa0-41e4-bd05-42d1fb030da5.png)

# Demo video

## We actually shot a video containing  cracks and processed it with our method. Here are the original video and the video processed with our method.

### original video

https://user-images.githubusercontent.com/123912496/217747704-e146532c-497e-4a42-8a77-a296317f0714.mp4

### Post-processing video

https://user-images.githubusercontent.com/123912496/217757140-69f430a2-aa3c-4b71-9d6b-acb4727e2441.mp4

