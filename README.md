**Predictive Modeling of CDK5 Inhibitors A Machine Learning Filtration Approach for Virtual Screening**

**Project Overview**

This project uses cheminformatics and machine learning to predict potential CDK5 inhibitors.
CDK5, a cyclin-dependent kinase, is involved in various neurodegenerative diseases, making it a significant target
for drug discovery. The project implements data retrieval, molecular feature extraction, and machine learning models
to classify and evaluate chemical compounds for their inhibitory potential.

-Installation Instructions

-------**Prerequisites**-------

Python 3.7+

pip for package management

-----------**Steps**-----------

Clone this repository or download the project files.

Navigate to the project directory.

Install dependencies from the requirements.txt file:

pip install -r requirements.txt

Ensure that joblib is installed for loading the pre-trained models.

--------**Files Included**-----

Predictive Modeling of CDK5 Inhibitors.py: Main script for training models and processing data and predicting new compound activities

combined_data.csv: Processed dataset combining ChEMBL and PubChem data.

processed_cdk5_data_combined_features.csv: Feature set after preprocessing.

rf_model_with_features.pkl and svm_model_with_features.pkl: Trained Random Forest and SVM models.

top_5_cdk5_inhibitors.csv: Output of top 5 predicted CDK5 inhibitors.

-----------**Usage**-----------

Training and Evaluation

Run the Predictive Modeling of CDK5 Inhibitors.py script to preprocess data, balance the dataset using SMOTE, train models,
and evaluate their performance and Predicting New Compounds


-----------**Output**----------

Classification metrics and ROC-AUC scores for both models.

Visualizations of class distributions and ROC curves.

CSV file containing top predicted CDK5 inhibitors.

---------**Dependencies**------

See requirements.txt for a detailed list of dependencies
