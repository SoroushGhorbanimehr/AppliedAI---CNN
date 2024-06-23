
# Scene Classification Using Decision Trees and Convolutional Neural Networks

## High-Level Description

This project explores scene classification using two main approaches: Decision Trees and Convolutional Neural Networks (CNNs). Initially, the study implements and evaluates Supervised and Semi-supervised Decision Trees. To improve performance, the project then advances to more sophisticated CNN models. The goal is to classify scenes into five distinct categories: Airport, Classroom, Music Studio, Bakery, and Bowling Alley. The performance of these models is evaluated using metrics such as accuracy, precision, recall, and F1 score.

## Requirements

To run this project, you need the following libraries:

1. Python 3.x
2. numpy
3. pandas
4. matplotlib
5. seaborn
6. scikit-learn
7. Pillow
8. torch
9. torchvision
10. tqdm
11. graphviz

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow torch torchvision tqdm graphviz google-colab
```

## Instructions
### Data Preparation
1- Upload the dataset to your Google Drive. Make sure the dataset is organized into categories such as Airport, Bakery, Bowling, Classroom, and Music Studio.
2- Update the paths in the provided script to point to your Google Drive directory.

### Running the Code
1- Open the provided code in Google Colab.
2- Mount your Google Drive in Colab using the following command:

```
from google.colab import drive
drive.mount('/content/drive')
```

3- Ensure the paths in the code point to the correct locations in your Google Drive. For example:

```
data_dir = '/content/drive/MyDrive/Your_Dataset_Path/'
output_dir = '/content/drive/MyDrive/Your_Output_Path/'
```

4- Run all cells in the Colab notebook. This will perform the following steps:
-   Extract images from the dataset and resize them to 224x224 pixels.
-   Split the data into training, validation, and test sets.
-   Train and validate supervised and semi-supervised Decision Tree models.
-   Train and validate CNN models.
-   Evaluate the models and output the results.

## Dataset

The dataset used in this project is the Places dataset provided by MIT CSAIL.

To obtain the dataset:

1- Go to the MIT CSAIL Places dataset download page(http://places2.csail.mit.edu/download.html).

2- Select the subset of the dataset that includes the desired categories (Airport, Classroom, Music studio, Bakery, Bowling).

3- Download the dataset and upload it to your Google Drive.

4- Update the paths in the provided script to point to your Google Drive directory.

## Models and Training

### Decision Trees

The project initially implements two types of Decision Tree models:

1.  **Supervised Decision Tree**: Trained using labeled data to classify images.
2.  **Semi-supervised Decision Tree**: Combines labeled and unlabeled data to improve classification performance.

### Convolutional Neural Networks

The project then advances to CNN models to improve classification accuracy:

1.  **SimpleCNN**: A basic CNN with two convolutional layers followed by max-pooling and fully connected layers.
2.  **CustomCNN**: An advanced CNN with more layers and dropout for better performance.
3.  **DeepCNN**: An advanced CNN with multiple convolutional layers, batch normalization, dropout, and fully connected layers.

#### Training and Validation

1.  **Data Augmentation and Preprocessing**:
    
    -   Apply data augmentation techniques such as random horizontal and vertical flips, random rotations, color jitter, random resized crops, and random affine transformations.
    -   Normalize the images with appropriate mean and standard deviation values.
2.  **Model Training**:
    
    -   Define the CNN architectures.
    -   Train the models using cross-entropy loss and optimizers like Adam or AdamW.
    -   Use learning rate scheduling and gradient scaling for efficient training.
    -   Train the models for a specified number of epochs, typically around 50 epochs.
3.  **Model Evaluation**:
    
    -   Evaluate the models on the validation set after each epoch.
    -   Save the best models based on validation accuracy.

### Testing the trained Model

1.  **Load the trained CNN Model weights**:
    
    -   Load the saved model weights from the specified file paths.
2.  **Run Inference on Test Dataset**:
    
    -   Evaluate the models on the test dataset to obtain accuracy, confusion matrix, and classification report.
    -   Plot confusion matrix and classification metrics such as precision, recall, and F1-score for each class.


## Instructions to Run the trained weights of our Model on a Sample Test Image

1.  **Load the Model and Make Predictions**:
    -   Load the weights of the model and make predictions on a single test image.
    -   Transform the image to the required input dimensions and normalize it.
    -   Print the predicted class for the given test image.

**Note** - Your image should be present in your Google Drive's "My Drive" directory and should have the '.jpg' extension. Once you've updated the same in Google Collab Notebook for the DeepCNN model, you would be able to predict the label for your test image.

## Obtaining the Dataset

1.  **Download the Dataset**:
    
    -   The dataset used in this project is assumed to be available in your Google Drive. If not, please reach out to one of the project contributors to get access. Alternatively, you can download the dataset from this [Google Drive Link](https://drive.google.com/drive/folders/1C4C3Xf7W2Z04zgQRpCr63w3JyYT42cKl?usp=drive_link) & upload it to your Google Drive.
2.  **Downloading the CNN Model**:
    
    -   You can download the **DeepCNN** model from this link [DeepCNN Model](https://drive.google.com/file/d/1UAcTgh4oT2fY9cSZT3WeqwkJgPobUXc6/view?usp=drive_link). Once downloaded, you should place this model in your **My Drive** directory of Google Drive.
3.  **Running the Script**:
    
    -   Once you have the dataset & CNN model in your Google Drive, you can run the Jupyter Notebook in your Google Collab Environment with selecting GPU as the compute environment.


## Contact
1- Sorush.gm@gmail.com

2- 

3-
