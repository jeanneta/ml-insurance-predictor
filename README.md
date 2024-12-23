# Predicting Customer Response for Car Insurance Policies

## Overview

This project focuses on building a Machine Learning model to predict whether a potential customer will purchase a car insurance policy offered by a sales representative. The model is trained using the provided dataset and stores the trained model as a pickle file for future use. The aim is to identify patterns in the data that influence customer decisions and make predictions efficiently.

## Task Description

The task involves:

1. Understanding and preprocessing the dataset to prepare it for machine learning.
2. Building a machine learning model to identify patterns and predict customer responses.
3. Saving the trained model using the pickle format.

## Dataset Information

The dataset includes the following features:

1. **id**: Unique identifier for each entry or customer in the dataset.
2. **Gender**: Gender of the customer ('Male' or 'Female').
3. **Age**: Age of the customer in years.
4. **Driving_License**: Driving license ownership status (0 for not owning, 1 for owning).
5. **Region_Code**: Geographic region code of the customer.
6. **Previously_Insured**: Whether the customer already has insurance (0 for no, 1 for yes).
7. **Vehicle_Age**: Age of the vehicle owned by the customer ('New', '1-2 Year', 'More than 2 Years').
8. **Vehicle_Damage**: Whether the customer’s vehicle has been previously damaged ('Yes' or 'No').
9. **Annual_Premium**: Annual premium the customer needs to pay for insurance.
10. **Policy_Sales_Channel**: Numerical code indicating the sales channel used (e.g., agent, online).
11. **Vintage**: Number of days since the customer has been associated with the insurance company.
12. **Response**: Customer’s response to the insurance offer (0 for not interested, 1 for interested).

## Steps Taken

### 1. Data Preprocessing

- **Handling Missing Values**: Checked and imputed any missing data to ensure completeness.
- **Feature Encoding**: Converted categorical variables (‘Gender’, ‘Vehicle_Age’, and ‘Vehicle_Damage’) into numerical formats suitable for the model.
- **Scaling**: Scaled numerical features such as ‘Age’, ‘Annual_Premium’, and ‘Vintage’ to standardize data.

### 2. Model Building

- **Algorithms Used**: Multiple models were implemented to evaluate their effectiveness for the task, including:
    - **Random Forest**: Achieved an accuracy of 87.5%, but struggled with predicting the minority class.
    - **XGBoost**: Also achieved an accuracy of 87.5%, with competitive results and faster training.
    - **KNN**: Included for comparison but showed poor performance, predicting only the majority class.
    - **SMOTE with Random Forest**: Improved recall for the minority class but resulted in a lower overall accuracy of 76%.
- **Performance Metrics**:
    - Accuracy: Overall correctness.
    - Precision, Recall, and F1-Score: Evaluated to measure performance on both classes.

### 3. Saving the Model

- Used Python’s `pickle` module to serialize the trained model into a `.pkl` file for easy deployment and reuse.

## Results and Discussion

- **Key Insights**:
    - Models achieved similar overall accuracy (~87.5%) for predicting customer responses.
    - **Random Forest** and **XGBoost** performed best overall, but both struggled with imbalanced data.
    - Applying **SMOTE** addressed class imbalance, significantly improving recall for the minority class while lowering accuracy.

## Best Model

**XGBoost** was chosen as the best model due to its:

- Efficient handling of large datasets.
- Competitive performance metrics.
- Suitability for imbalanced datasets with minor parameter tuning.

## How to Use

1. Clone the repository:
    
    ```bash
    git clone https://github.com/jeanneta/ml-insurance-predictor.git
    ```
    
2. Run the Jupyter Notebook:
    
    ```bash
    jupyter notebook app.ipynb
    
    ```
    
3. Load and use the trained model:
    
    ```python
    import pickle
    with open('modelKNN.pkl', 'rb') as file: 
        model = pickle.load(file)
    prediction = model.predict(new_data)
    
    ```
    

## Future Work

- Fine-tune hyperparameters for Random Forest and XGBoost to improve predictions further.
- Explore additional algorithms or ensemble methods.
- Implement additional preprocessing techniques to handle imbalanced datasets without relying solely on SMOTE.
- Develop a deployment pipeline for real-world usage.