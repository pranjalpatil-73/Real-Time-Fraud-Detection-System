# Real-Time-Fraud-Detection-System

Real-Time Fraud Detection in Financial Transactions

This project develops a machine learning-driven system for the real-time detection of fraudulent financial transactions. Leveraging historical transaction data, the system aims to identify suspicious activities with high accuracy, enabling financial institutions to mitigate risks, prevent losses, and maintain trust.

## Importance of Solving This Problem

Fraud in financial transactions poses significant threats to individuals and institutions alike. Addressing this problem is paramount due to:

* **Financial Loss Prevention**: Fraudulent activities result in direct monetary losses for banks, businesses, and consumers. Effective real-time detection minimizes these losses.
* **Reputation and Trust**: Frequent fraud incidents can severely damage a financial institution's reputation, eroding customer trust and leading to account closures or customer attrition.
* **Regulatory Compliance**: Financial regulators impose stringent requirements on institutions to implement robust anti-fraud measures. Failure to comply can lead to hefty fines and legal repercussions.
* **Customer Security**: Protecting customers from unauthorized transactions enhances their confidence in the financial system and the services provided.
* **Operational Efficiency**: Automated fraud detection systems reduce the need for manual review of every transaction, thereby streamlining operations and reducing human error.
* **Evolving Threat Landscape**: Fraud schemes are constantly evolving. A real-time, adaptive detection system is crucial to staying ahead of sophisticated fraudsters.

## Features

* **Data Loading and Initial Exploration**: Imports and provides an initial overview of the synthetic financial transaction dataset, including its structure, data types, and basic statistics.
* **Comprehensive Data Preprocessing**:

  * Handles potential missing values in numerical and categorical features.
  * Performs feature scaling using `StandardScaler` on numerical attributes.
  * Applies `OneHotEncoder` for categorical feature transformation, ensuring compatibility with machine learning models.
  * Creates time-based features (e.g., hour, day\_of\_week) from transaction timestamps to capture temporal patterns.
* **Handling Imbalanced Data**: Uses oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
* **Machine Learning Model Training**:

  * Trains and evaluates Logistic Regression and Random Forest Classifier.
  * Uses GridSearchCV for hyperparameter tuning.
* **Robust Model Evaluation**:

  * Evaluates using Precision, Recall, F1-score, ROC AUC score, Confusion Matrix.
  * Visualizes Precision-Recall Curves.
* **Model Persistence**: Saves the best-performing model using `pickle`.
* **Real-Time API Simulation**:

  * A `FraudDetectionAPI` class to simulate real-time prediction.
  * Loads a trained model and preprocessing pipeline.
  * Processes single transactions and logs predictions.
  * Provides live fraud statistics.

## Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Scikit-learn**: Includes tools for data splitting, preprocessing, training, evaluation
* **imbalanced-learn (imblearn)**: SMOTE for balancing datasets
* **Pickle**: For saving/loading models

## Getting Started

### Prerequisites

* Python 3.7+
* Jupyter Notebook or JupyterLab

### Installation

1. Download the notebook `Real-Time Fraud Detection in Financial Transactions19.ipynb`
2. Place the dataset (e.g., `fraud_transactions.csv`) in the same directory
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook "Real-Time Fraud Detection in Financial Transactions19.ipynb"
```

2. Run all cells sequentially:

   * Loads and preprocesses the data
   * Trains and evaluates models
   * Saves the model as a `.pkl` file
   * Defines and tests the `FraudDetectionAPI`

#### Real-Time Fraud Prediction Example:

```python
# api = FraudDetectionAPI('fraud_detection_model.pkl')
# transaction_example = df.iloc[0].to_dict()
# response = api.process_transaction(transaction_example)
# print("Transaction Processing Response:", response)
```

#### Get Fraud Statistics Example:

```python
# stats = api.get_fraud_stats(last_n=50)
# print("Recent Fraud Statistics:", stats)
```

## Future Enhancements

* **Production Deployment**: Build a web service using Flask/FastAPI
* **Feedback Loop Integration**: Use new labeled data for retraining
* **Anomaly Detection**: Add unsupervised methods like Isolation Forest or Autoencoders
* **Graph Neural Networks**: Use GNNs for analyzing transaction relationships
* **Explainable AI (XAI)**: Integrate SHAP or LIME for interpretability
* **Streaming Data Integration**: Connect with Kafka or Flink for real-time streams
