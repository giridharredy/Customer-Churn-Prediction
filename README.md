# Customer Churn Prediction using Logistic Regression

## Overview
This project predicts customer churn (whether a customer will leave a service) using Logistic Regression. The dataset used is the **Telco Customer Churn dataset**, which includes features like tenure, monthly charges, and total charges.

## Dataset
- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Features**: 
  - `tenure`: Number of months the customer has stayed with the company.
  - `MonthlyCharges`: Monthly subscription charges.
  - `TotalCharges`: Total charges incurred by the customer.
  - Other categorical features: `gender`, `Partner`, `Dependents`, etc.
- **Target Variable**: `Churn` (Yes/No).

## Dependencies
To run this project, you need the following Python libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn (for SMOTE)
- matplotlib (for visualization)
- seaborn (for visualization)

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
Steps to Run the Code

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
Download the dataset from Kaggle and place it in the project folder.

Run the Jupyter Notebook or Python script:

```bash

jupyter notebook customer_churn_prediction.ipynb
```
or

```bash

python customer_churn_prediction.py
```
Code Structure
Data Preprocessing: Handling missing values, encoding categorical features, and scaling numerical features.

Model Training: Training a Logistic Regression model.

Model Evaluation: Evaluating the model using accuracy, confusion matrix, and classification report.

Prediction: Making predictions on new data.

Results
Accuracy: [Insert accuracy score]

Confusion Matrix:

[[TN, FP],
 [FN, TP]]

 
Classification Report:

Precision: [Value]

Recall: [Value]

F1-Score: [Value]

Future Improvements
Try advanced models like Random Forests or XGBoost.

Perform hyperparameter tuning for better performance.

Handle class imbalance using techniques like SMOTE or class weights.

License
This project is licensed under the MIT License. See the LICENSE file for details.

---
