# Salary_Prediction_Using_Machine_Learning
This project focuses on building a machine learning model to predict the "Expected CTC" (Cost to Company) for prospective employees. The goal is to create a robust system that minimizes human bias in salary decisions, ensuring fair compensation for candidates with similar profiles.

📋 Dataset

The primary dataset used for this project is expected_ctc.csv. This file contains various professional and educational details about past applicants, including their experience, previous salary, education level, and other qualifications. This historical data serves as the foundation for training our predictive models.

🚀 Project Workflow

I followed a structured approach to build and evaluate the models, ensuring the data was properly cleaned and prepared before training.

    Data Cleaning and Preparation: I started by loading the dataset and cleaning the column names to prevent errors. A key decision was to handle missing values by filling them in rather than deleting data; I used the mode for categorical columns and the median for numerical ones. I also removed non-predictive columns like Applicant_ID.

    Exploratory Data Analysis (EDA): I dove into the data to understand its characteristics. By creating histograms and statistical summaries, I analyzed the distribution of key features like Current_CTC and Total_Experience, which helped inform my modeling choices.

    Feature Engineering: The models need numerical input, so I converted all categorical (text-based) columns like Department and Role into a numerical format using one-hot encoding. This created new binary columns that the models could understand.

    Outlier Analysis: I performed an analysis to check for extreme values (outliers) using the 1.5 * IQR statistical rule. The analysis showed that the dataset, after the imputation step, did not contain any outliers according to this method.

    Model Training and Evaluation: I split the data into an 80% training set and a 20% testing set. After scaling the features, I trained five different regression models on the training data and then evaluated their performance on the unseen testing data to see how accurately they could predict the Expected_CTC.

🤖 Models Used

I implemented a suite of five common and effective supervised learning models for this regression task:

    Linear Regression: A baseline model to understand linear relationships.

    Decision Tree Regressor: A model that captures non-linear patterns.

    Random Forest Regressor: An advanced ensemble model that combines many decision trees for higher accuracy.

    K-Nearest Neighbors (KNN): A distance-based algorithm that makes predictions based on similar data points.

    Support Vector Regressor (SVR): A powerful model effective in high-dimensional spaces.

📊 Results and Observations

After training and evaluating all five models, the results were clear.

The Random Forest Regressor consistently outperformed the other models. It achieved the highest R-squared (R²) score, indicating it was the most accurate in explaining the variance in the expected salary. It also had the lowest Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), confirming its superior predictive power. This result is expected, as Random Forest models are excellent at handling complex, tabular data like this.

⚙️ How to Run the Project

To run this project yourself, you will need to have Python and the necessary libraries installed.

    Place the expected_ctc.csv file in the same directory as the Python script or notebook.

    Run the script. It will execute all the steps from data cleaning to model evaluation and will output the performance metrics and visualizations.

Libraries Used

Make sure you have the following libraries installed:

    pandas

    numpy

    matplotlib

    seaborn

    scikit-learn

You can install them using pip:
pip install pandas numpy matplotlib seaborn scikit-learn
