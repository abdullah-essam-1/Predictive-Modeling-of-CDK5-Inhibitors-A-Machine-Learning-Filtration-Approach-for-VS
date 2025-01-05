Project Architecture
Modules and Scripts
1.	Predictive Modeling of CDK5 Inhibitors.py: Contains data preprocessing, model training, evaluation, and visualization, handles prediction of activity for new compounds using pre-trained models.
2.	refine_structure: A function for refining molecular structures by removing salts, neutralizing charges, and optimizing geometries.
3.	compute_fingerprints: Generates Morgan fingerprints for molecular similarity analysis.
4.	compute_geometrical_chemical_features: Computes molecular descriptors to use as additional features.
Design Decisions
•	Data Sources: ChEMBL and PubChem databases are used to obtain CDK5-related bioactivity data.
•	Feature Engineering: Morgan fingerprints and geometric-chemical descriptors are combined for robust feature representation.
•	Imbalanced Data Handling: SMOTE is employed to address class imbalance.
Algorithms Used
•	Random Forest Classifier: Used for robust feature handling and ensemble learning.
•	Support Vector Machine (SVM): Applied for classification with kernel trick optimization.
•	GridSearchCV: Hyperparameter tuning for optimal model performance.
Model Evaluation
•	Classification reports include precision, recall, and F1-score.
•	ROC-AUC score for evaluating binary classification effectiveness.
Dependencies
•	See requirements.txt for specific library versions.
