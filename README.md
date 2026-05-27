
### Project Overview

Customer churn occurs when customers stop using a company’s services, leading to revenue loss and reduced business growth. Predicting customer churn helps businesses understand customer behavior and take proactive actions to improve customer retention.

This project focuses on analyzing and predicting customer churn using the Telco Customer Churn dataset. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model training, and performance evaluation.

The goal of this project is to identify patterns that influence customer churn and build a machine learning model capable of predicting whether a customer is likely to churn or not.

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest Classifier
- Jupyter Notebook

### Dataset Information

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

### Business Problem

Customer churn negatively impacts business revenue and customer retention. This project aims to:

- Identify factors affecting customer churn
- Analyze churn behavior
- Predict customer churn using machine learning
- Help businesses improve customer retention strategies

### Project Workflow

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

### Exploratory Data Analysis (EDA)

The project includes:

- Churn distribution analysis
- Customer behavior analysis
- Feature correlation analysis
- Service-based churn patterns
- Visual analysis using count plots and charts
- 
### EDA Process :-

### 1. Importing Libraries and Dataset

### 1.1 Loading the Dataset
We start by importing the necessary Python libraries and loading the Telco Customer Churn dataset. This dataset contains various customer details such as service plans, usage behavior and churn status. 
Code: import numpy as np
import pandas as pd

dataset = pd.read_csv('/filename')

dataset.head()


### 1.2 Understanding the Dataset
To gain insights into the dataset we first check for missing values and understand its structure. The dataset includes features such as:

#### Code: print(dataset.isnull().sum())
print(dataset.describe())

#### Output: I have uploaded the output in the form of Snapshot for clear understanding.

tenure – The number of months a customer has stayed with the company.
InternetService – The type of internet service the customer has DSL, Fiber optic or None.
PaymentMethod– The method the customer uses for payments.
Churn – The target variable i.e Yes for customer churned and No for customer stayed.


### 1.3 Analyzing Churn Distribution
We check the number of churners and non-churners to understand the balance of the dataset.

#### Code: import seaborn as sns
import matplotlib.pyplot as plt

print(dataset['Churn'].value_counts())
sns.countplot(x='Churn', data=dataset, palette='coolwarm')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

Output: Churn Distribution chart ( i Uploaded in form snapshot )

### 2. Data Preprocessing

#### 2.1 Handling Missing and Incorrect Values
Before processing we ensure that all numerical columns contain valid values. The TotalCharges column sometimes has empty spaces which need to be converted to numerical values.

pd.to_numeric(dataset['TotalCharges'], errors='coerce') converts the TotalCharges column to numerical format. If any value is not convertible (e.g., empty spaces), it replaces it with NaN.

.fillna(dataset['TotalCharges'].median(), inplace=True) replaces missing values (NaN) with the median of the column to maintain consistency in numerical values.

Code: dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)


#### 2.2 Handling Categorical Variables

Some features like State, International Plan and Voice Mail Plan are categorical and must be converted into numerical values for model training.

LabelEncoder() converts categorical values into numerical form. Each unique category is assigned a numeric label.

The loop iterates through each categorical column and applies fit_transform() to encode categorical variables into numbers.

Code: from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col])
    

#### 2.3 Feature Selection and Splitting Data
We separate the features (X) and target variable (y) and split the dataset into training and testing sets.

X = dataset.drop(['customerID', 'Churn'], axis=1) removes the customerID (irrelevant for prediction) and Churn column (target variable).

y = dataset['Churn'] defines y as the target variable, which we want to predict.
train_test_split() splits data into 80% training and 20% testing for model evaluation.

Code: from sklearn.model_selection import train_test_split

X = dataset.drop(['customerID', 'Churn'], axis=1)

y = dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#### 2.4 Feature Scaling
Since features are on different scales we apply standardization to improve model performance. It prevents models from being biased toward larger numerical values and improves convergence speed in optimization algorithms like gradient descent

StandardScaler(): Standardizes data by transforming it to have a mean of 0 and a standard deviation of 1 ensuring all features are on a similar scale.
fit_transform(X_train): Fits the scaler to the training data and transforms it.
transform(X_test): Transforms the test data using the same scaling parameters.

#### Code: from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


### 3.  Model Training and Prediction
For training our model we use Random Forest Classifier. It is an ensemble learning method that combines the results of multiple decision trees to make a final prediction.

#### Code: from sklearn.ensemble import RandomForestClassifier

clf = from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


#### Output:

randomforestclassifier()

### 4. Model Evaluation

The model performance was evaluated using:

 #### 4.1 Accuracy Score
  from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

#### Output:
Model Accuracy: 0.78


#### 4.2 Confusion Matrix and Performance Metrics
We evaluate precision, recall and accuracy using a confusion matrix.

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix')
plt.show()

#### Output: 
Confusion matrix shows how well the model predicts customer churn. It correctly identifies 924 non-churners and 181 churners. However 117 non-churners are wrongly classified as churners and 187 churners are missed. The high number of missed churners suggests the model may need further tuning.



#### Key Insights

- Customers with shorter tenure are more likely to churn.
- Contract type significantly impacts customer retention.
- Customers using fiber optic internet services show higher churn rates.
- Electronic payment methods are associated with higher churn probability.
- Customer support and additional services influence retention behavior.

#### Visualizations Included

The project includes the following visualizations:

- Churn Distribution Plot
- Count Plots
- Correlation Analysis
- Confusion Matrix
- Feature-Based Customer Analysis

#### Learning Outcomes

Through this project, I learned:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Label Encoding
- Feature Scaling
- Machine Learning Model Training
- Model Evaluation Techniques
- Customer Retention Analytics

#### Future Improvements

Future enhancements for this project may include:

- Hyperparameter Tuning
- XGBoost and Advanced Models
- Customer Segmentation
- Real-time Churn Prediction Dashboard
- Deep Learning Models
- Deployment using Flask or Streamlit

#### Project Structure

├── Customer_Churn_Analysis.ipynb
├── Telco-Customer-Churn.csv
├── README.md



#### Conclusion

This project demonstrates how machine learning and data analysis techniques can be used to predict customer churn and understand customer behavior.

The project highlights the importance of customer retention analytics and predictive modeling in helping businesses reduce churn and improve long-term customer engagement.
