# 102-flowers-deep-learning-pytorch-with-VGG
An extensive, step-by-step tutorial for novices using PyTorch for computer vision and deep learning

## Overview
This project serves as a comprehensive guide for beginners in computer vision and deep learning using PyTorch. The focus is on utilizing the 102-flowers dataset to perform straightforward image classification with a pre-trained VGG19 backbone.

### Project Highlights
1. **Transfer Learning Philosophy:** Employing transfer learning to address the 102-class classification problem. The project uses an ImageNet pre-trained VGG19 feature extraction head without fine-tuning. A new classifier head is trained specifically for the 102-flowers dataset.

2. **Modular Components:** The project includes distinct components for network training and validation, testing, checkpointing, predicting with single new images, as well as image pre-processing and displaying. Each of these functionalities is encapsulated in self-contained functions, enhancing reusability.

## Why Engage in this Project
This project is designed as a tutorial-like experience, carefully crafted for beginners. It offers a step-by-step walkthrough, making it accessible for those who are new to computer vision and deep learning concepts.

## Instructions for Using the Repository
- **PyTorch Version:** The code is implemented using PyTorch 0.4.1 with Python 3.7 and Cuda 9.0. Special utilities for switching between CPU and GPU are included.

- **Dataset Download:** Obtain the 102-flowers dataset from [this link](https://www.kaggle.com/wassimseifeddine/102flowersdataset/downloads/102flowersdataset.zip/1). Once downloaded, unzip the file into your project directory, ensuring it is named './flowers'. The directory structure should include '/train', '/valid', and '/test' subdirectories within './flowers'.

- **Code Structure:** The functions for training, testing, prediction, checkpoint saving and loading, as well as image pre-processing, are encapsulated for easy reuse.

- **File Formats:** Both a Jupyter Notebook version and a plain Python (.py) file version are provided for your convenience.

## Additional Handbook of Tips
- **Jupyter Notebook Tips:** A collection of tips for absolute newcomers to Jupyter notebooks, aimed at simplifying the learning curve.

- **Linux OS & PyTorch Tips:** An ongoing compilation of tips for those new to Linux OS and PyTorch, sharing insights that have proven beneficial.

Feel free to explore, learn, and enjoy the journey into the world of computer vision and deep learning with PyTorch!

The training batchsize is 32.
Epoch: 1/10, itrs: 40, Train_loss: 5.5381, Valid_loss: 2.7243, Valid_Acc: 0.3762
Epoch: 1/10, itrs: 80, Train_loss: 3.0094, Valid_loss: 1.7837, Valid_Acc: 0.5553
Epoch: 1/10, itrs: 120, Train_loss: 2.6317, Valid_loss: 1.5662, Valid_Acc: 0.5913
Epoch: 1/10, itrs: 160, Train_loss: 2.4557, Valid_loss: 1.3323, Valid_Acc: 0.6671
Epoch: 1/10, itrs: 200, Train_loss: 2.1837, Valid_loss: 1.2129, Valid_Acc: 0.6803
Epoch 1 takes 150.9493 sec
Epoch: 2/10, itrs: 40, Train_loss: 1.9879, Valid_loss: 1.0824, Valid_Acc: 0.7055
Epoch: 2/10, itrs: 80, Train_loss: 1.9305, Valid_loss: 0.9302, Valid_Acc: 0.7536
Epoch: 2/10, itrs: 120, Train_loss: 1.9148, Valid_loss: 1.0403, Valid_Acc: 0.7175
Epoch: 2/10, itrs: 160, Train_loss: 1.8907, Valid_loss: 0.9395, Valid_Acc: 0.7584
Epoch: 2/10, itrs: 200, Train_loss: 1.9008, Valid_loss: 0.8237, Valid_Acc: 0.7873
Epoch 2 takes 302.8089 sec
Epoch: 3/10, itrs: 40, Train_loss: 1.7782, Valid_loss: 1.0154, Valid_Acc: 0.7524
Epoch: 3/10, itrs: 80, Train_loss: 1.7307, Valid_loss: 0.8328, Valid_Acc: 0.7704
Epoch: 3/10, itrs: 120, Train_loss: 1.7348, Valid_loss: 0.8179, Valid_Acc: 0.7849
Epoch: 3/10, itrs: 160, Train_loss: 1.7603, Valid_loss: 0.7925, Valid_Acc: 0.7897
Epoch: 3/10, itrs: 200, Train_loss: 1.7779, Valid_loss: 0.7266, Valid_Acc: 0.8077
Epoch 3 takes 454.6706 sec
Epoch: 4/10, itrs: 40, Train_loss: 1.5578, Valid_loss: 0.7834, Valid_Acc: 0.7945
Epoch: 4/10, itrs: 80, Train_loss: 1.6276, Valid_loss: 0.7098, Valid_Acc: 0.8089
Epoch: 4/10, itrs: 120, Train_loss: 1.681, Valid_loss: 0.6933, Valid_Acc: 0.8221
Epoch: 4/10, itrs: 160, Train_loss: 1.5898, Valid_loss: 0.7745, Valid_Acc: 0.8053
Epoch: 4/10, itrs: 200, Train_loss: 1.5475, Valid_loss: 0.7037, Valid_Acc: 0.8125
Epoch 4 takes 606.6841 sec
Epoch: 5/10, itrs: 40, Train_loss: 1.476, Valid_loss: 0.6839, Valid_Acc: 0.8173
Epoch: 5/10, itrs: 80, Train_loss: 1.546, Valid_loss: 0.7475, Valid_Acc: 0.7993
Epoch: 5/10, itrs: 120, Train_loss: 1.6268, Valid_loss: 0.7059, Valid_Acc: 0.8161
Epoch: 5/10, itrs: 160, Train_loss: 1.4951, Valid_loss: 0.7428, Valid_Acc: 0.8053
Epoch: 5/10, itrs: 200, Train_loss: 1.6101, Valid_loss: 0.6682, Valid_Acc: 0.8221
Epoch 5 takes 758.7217 sec
Epoch: 6/10, itrs: 40, Train_loss: 1.461, Valid_loss: 0.647, Valid_Acc: 0.8257
Epoch: 6/10, itrs: 80, Train_loss: 1.4419, Valid_loss: 0.703, Valid_Acc: 0.8125
Epoch: 6/10, itrs: 120, Train_loss: 1.5055, Valid_loss: 0.6625, Valid_Acc: 0.8281
Epoch: 6/10, itrs: 160, Train_loss: 1.505, Valid_loss: 0.6944, Valid_Acc: 0.8113
Epoch: 6/10, itrs: 200, Train_loss: 1.5086, Valid_loss: 0.6845, Valid_Acc: 0.8161
Epoch 6 takes 910.9159 sec
Epoch: 7/10, itrs: 40, Train_loss: 1.3695, Valid_loss: 0.5905, Valid_Acc: 0.8365
Epoch: 7/10, itrs: 80, Train_loss: 1.5547, Valid_loss: 0.6426, Valid_Acc: 0.8221
Epoch: 7/10, itrs: 120, Train_loss: 1.4563, Valid_loss: 0.5978, Valid_Acc: 0.8438
Epoch: 7/10, itrs: 160, Train_loss: 1.5062, Valid_loss: 0.611, Valid_Acc: 0.8389
Epoch: 7/10, itrs: 200, Train_loss: 1.5091, Valid_loss: 0.6734, Valid_Acc: 0.8341
Epoch 7 takes 1062.8523 sec
Epoch: 8/10, itrs: 40, Train_loss: 1.4018, Valid_loss: 0.6842, Valid_Acc: 0.8341
Epoch: 8/10, itrs: 80, Train_loss: 1.5311, Valid_loss: 0.6289, Valid_Acc: 0.8425
Epoch: 8/10, itrs: 120, Train_loss: 1.5183, Valid_loss: 0.6772, Valid_Acc: 0.8257
Epoch: 8/10, itrs: 160, Train_loss: 1.5935, Valid_loss: 0.6305, Valid_Acc: 0.8341
Epoch: 8/10, itrs: 200, Train_loss: 1.5654, Valid_loss: 0.6736, Valid_Acc: 0.8341
Epoch 8 takes 1214.7187 sec
Epoch: 9/10, itrs: 40, Train_loss: 1.5134, Valid_loss: 0.612, Valid_Acc: 0.8474
Epoch: 9/10, itrs: 80, Train_loss: 1.4792, Valid_loss: 0.6425, Valid_Acc: 0.8377
Epoch: 9/10, itrs: 120, Train_loss: 1.4268, Valid_loss: 0.6096, Valid_Acc: 0.8534
Epoch: 9/10, itrs: 160, Train_loss: 1.5247, Valid_loss: 0.5834, Valid_Acc: 0.851
Epoch: 9/10, itrs: 200, Train_loss: 1.4242, Valid_loss: 0.6043, Valid_Acc: 0.8305
Epoch 9 takes 1366.7112 sec
Epoch: 10/10, itrs: 40, Train_loss: 1.4581, Valid_loss: 0.6477, Valid_Acc: 0.8534
Epoch: 10/10, itrs: 80, Train_loss: 1.5092, Valid_loss: 0.6321, Valid_Acc: 0.8245
Epoch: 10/10, itrs: 120, Train_loss: 1.4665, Valid_loss: 0.5864, Valid_Acc: 0.8522
Epoch: 10/10, itrs: 160, Train_loss: 1.3373, Valid_loss: 0.5971, Valid_Acc: 0.851
Epoch: 10/10, itrs: 200, Train_loss: 1.4678, Valid_loss: 0.5359, Valid_Acc: 0.857
Epoch 10 takes 1518.7065 sec
# INFERENCE Stage
(1) IMAGE PREPROCESSING: Pre-proccess 1 image according to the format that recquried by the Network <br>
(2) Display the preprocessed image <br>
(3) Perform prediction on this image (by forward-pass it through the trained or loaded network) <br>
(4) Sanity Checking

![image](https://github.com/KAMAlhameedawi/102-flowers-deep-learning-pytorch-with-VGG/assets/149914341/b56420f8-c02f-4490-ba48-39df8bb4d44f)

[0.9938846826553345, 0.00579900061711669, 0.00016869889805093408, 0.00012347089068498462, 2.1379877580329776e-05]
{'15', '83', '84', '42', '46'}
![image](https://github.com/KAMAlhameedawi/102-flowers-deep-learning-pytorch-with-VGG/assets/149914341/e3792cb6-3851-4764-972d-8ee74ce603ce)
