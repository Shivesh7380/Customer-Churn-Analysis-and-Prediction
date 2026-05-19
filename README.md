
## Project Overview

Customer churn occurs when customers stop using a company’s services, leading to revenue loss and reduced business growth. Predicting customer churn helps businesses understand customer behavior and take proactive actions to improve customer retention.

This project focuses on analyzing and predicting customer churn using the Telco Customer Churn dataset. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model training, and performance evaluation.

The goal of this project is to identify patterns that influence customer churn and build a machine learning model capable of predicting whether a customer is likely to churn or not.

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest Classifier
- Jupyter Notebook

# Dataset Information

The dataset contains customer-related information such as:

- Customer ID
- Gender
- Internet Service
- Payment Method
- Contract Type
- Monthly Charges
- Total Charges
- Tenure
- Online Services
- Tech Support
- Streaming Services
- Churn Status

Dataset Format: CSV

# Business Problem

Customer churn negatively impacts business revenue and customer retention. This project aims to:

- Identify factors affecting customer churn
- Analyze churn behavior
- Predict customer churn using machine learning
- Help businesses improve customer retention strategies

# Project Workflow

1. Importing Libraries and Dataset
2. Understanding Dataset Structure
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Handling Missing Values
6. Encoding Categorical Variables
7. Feature Scaling
8. Model Training
9. Prediction and Evaluation
10. Business Insights Generation

# Data Preprocessing

The following preprocessing techniques were applied:

- Converted TotalCharges column into numerical format
- Handled missing values using median imputation
- Encoded categorical variables using LabelEncoder
- Removed irrelevant columns such as customerID
- Standardized features using StandardScaler

Example:

python
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)

# Exploratory Data Analysis (EDA)

The project includes:

- Churn distribution analysis
- Customer behavior analysis
- Feature correlation analysis
- Service-based churn patterns
- Visual analysis using count plots and charts

# Machine Learning Model

The project uses the following machine learning algorithm:

##  Model Training and Prediction
For training our model we use Random Forest Classifier. It is an ensemble learning method that combines the results of multiple decision trees to make a final prediction.

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

Output:

randomforestclassifier()

# Model Evaluation

The model performance was evaluated using:

 ## Accuracy Score
  from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

Output:
Model Accuracy: 0.78


## Confusion Matrix and Performance Metrics
We evaluate precision, recall and accuracy using a confusion matrix.

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix')
plt.show()

Output: Confusion matrix shows how well the model predicts customer churn. It correctly identifies 924 non-churners and 181 churners. However 117 non-churners are wrongly classified as churners and 187 churners are missed. The high number of missed churners suggests the model may need further tuning.



# Key Insights

- Customers with shorter tenure are more likely to churn.
- Contract type significantly impacts customer retention.
- Customers using fiber optic internet services show higher churn rates.
- Electronic payment methods are associated with higher churn probability.
- Customer support and additional services influence retention behavior.

# Visualizations Included

The project includes the following visualizations:

- Churn Distribution Plot
- Count Plots
- Correlation Analysis
- Confusion Matrix
- Feature-Based Customer Analysis

# Learning Outcomes

Through this project, I learned:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Label Encoding
- Feature Scaling
- Machine Learning Model Training
- Model Evaluation Techniques
- Customer Retention Analytics

# Future Improvements

Future enhancements for this project may include:

- Hyperparameter Tuning
- XGBoost and Advanced Models
- Customer Segmentation
- Real-time Churn Prediction Dashboard
- Deep Learning Models
- Deployment using Flask or Streamlit

# Project Structure

├── Customer_Churn_Analysis.ipynb
├── Telco-Customer-Churn.csv
├── README.md



# Conclusion

This project demonstrates how machine learning and data analysis techniques can be used to predict customer churn and understand customer behavior.

The project highlights the importance of customer retention analytics and predictive modeling in helping businesses reduce churn and improve long-term customer engagement.
