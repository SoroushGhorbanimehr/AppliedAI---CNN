# Scene Classification Using Decision Trees
This project is a comparative study of scene classification using Decision Trees. The aim is to classify scenes into five distinct categories: Airport, School, Hospital, Bakery, and Bowling using the Places dataset provided by MIT CSAIL. The study implements and evaluates three models: Supervised Decision Tree, Semi-supervised Decision Tree, and an iterative semi-supervised Decision Tree. The performance of these models is evaluated using metrics such as accuracy, precision, recall, and F1 score.

## Requirements
To run this project, you need the following libraries:

```
1- Python 3.x
2- numpy
3- pandas
4- matplotlib
5- seaborn
6- scikit-learn
7- Pillow
8- graphviz
9- tqdm
```

You can install the required libraries using the following command:
```
pip install numpy pandas matplotlib seaborn scikit-learn pillow tqdm graphviz google-colab
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

* Extract images from the dataset and resize them to 224x224 pixels.
* Split the data into training, validation, and test sets.
* Train and validate supervised and semi-supervised Decision Tree models.
* Evaluate the models and output the results.



### Source Code

The source code is available in the IPYNB file, which includes the implementation of supervised and semi-supervised Decision Tree models.

## Dataset

The dataset used in this project is the Places dataset provided by MIT CSAIL.

To obtain the dataset:

1- Go to the MIT CSAIL Places dataset download page(http://places2.csail.mit.edu/download.html).

2- Select the subset of the dataset that includes the desired categories (Airport, School, Hospital, Bakery, Bowling).

3- Download the dataset and upload it to your Google Drive.

4- Update the paths in the provided script to point to your Google Drive directory.





