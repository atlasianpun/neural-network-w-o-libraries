# README

## Project Title: Neural Network Implementation for Iris Dataset Classification

### Overview
This project implements a neural network from scratch for classifying the Iris dataset using Python. The dataset is loaded from Google Drive, preprocessed, and used to train and evaluate a neural network. Hyperparameters are optimized using the Hyperopt library.

### Requirements
- Python 3.x
- Google Colab
- Libraries: pandas, numpy, scikit-learn, plotly, matplotlib, seaborn, scipy, hyperopt

### Dataset
Download the Iris dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/iris) and save it to your Google Drive.

### Instructions
1. **Load the Dataset:** Place the dataset in your Google Drive at `/content/drive/MyDrive/ML/Iris.csv`.
2. **Adjust File Path:** Update the `file_path` variable in the code to point to the dataset's location.
3. **Run the Code:** Execute the notebook or script in Google Colab by selecting `Runtime` > `Run all`.

### Notes
- The neural network is implemented from scratch without using any specialized neural network libraries.
- Different activation functions (`relu`, `sigmoid`, `tanh`) are available.
- Hyperparameter optimization is performed using Hyperopt to improve model accuracy.

For any issues or questions, feel free to reach out to the project maintainers.
