
# Heart Disease Prediction

This project uses machine learning techniques to predict the risk of heart disease in patients. It is designed for healthcare professionals, researchers, and data scientists interested in early detection and risk assessment of heart disease.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Heart disease is a leading cause of mortality worldwide. Early prediction can help in timely intervention and improved patient outcomes. This project employs machine learning models to classify patients as high or low risk based on their clinical features.

## Dataset

The dataset used for this project consists of various medical parameters of patients such as:

- Age
- Gender
- Blood pressure
- Cholesterol levels
- Chest pain type
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression (oldpeak)
- The number of major vessels colored by fluoroscopy
- Thalassemia

*Note: Please refer to the source dataset for detailed attribute descriptions.*

## Features

- Data exploration and visualization
- Data preprocessing and cleaning
- Model training and evaluation (using algorithms such as Logistic Regression, Decision Trees, Random Forest, SVM, etc.)
- Performance metrics (accuracy, confusion matrix, ROC curve)
- Predictive function for new data

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Sriksgame/Heart-disease.git
   cd Heart-disease
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset and update the data paths if necessary.
2. Run the main script to train and evaluate the model:
   ```bash
   python main.py
   ```
3. Predict risk for new patient data using the provided functions or scripts.

## Model Details

This project includes experimentation with multiple models. You can modify `main.py` to try different algorithms and tune hyperparameters for optimal performance.

## Results

*Summary of model performance will be updated here after running experiments.*

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

## License

This project is licensed under the MIT License.

---

*For questions or collaboration, contact [Sriksgame](https://github.com/Sriksgame).*
