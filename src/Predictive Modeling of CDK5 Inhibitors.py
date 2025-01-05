import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from pubchempy import get_compounds
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Helper functions
salt_remover = SaltRemover()
neutralizer = rdMolStandardize.Uncharger()

def refine_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = salt_remover.StripMol(mol)
        mol = neutralizer.uncharge(mol)
        if Descriptors.MolWt(mol) > 100 and mol.GetNumAtoms() > 3:
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol)
            except:
                return None
            return mol
    return None

def refine_and_filter_data(data):
    valid_smiles = []
    refined_mols = []
    for smi in data['SMILES']:
        mol = refine_structure(smi)
        if mol:
            valid_smiles.append(smi)
            refined_mols.append(mol)
    return valid_smiles, refined_mols

def compute_fingerprints(mol_list):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fps = [np.array(morgan_gen.GetFingerprintAsNumPy(mol)) for mol in mol_list]
    return np.array(fps)

def compute_geometrical_chemical_features(mol_list):
    descriptor_names = [
        'MolWt', 'MolLogP', 'MolMR', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors', 'TPSA',
        'BalabanJ', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
        'Kappa1', 'Kappa2', 'Kappa3', 'FractionCSP3', 'HeavyAtomCount',
        'RingCount', 'NHOHCount', 'NOCount'
    ]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    features = []
    for mol in mol_list:
        try:
            features.append(calc.CalcDescriptors(mol))
        except:
            features.append([None] * len(descriptor_names))
    return pd.DataFrame(features, columns=descriptor_names)

# Retrieve ChEMBL data
target = new_client.target
cdk5 = target.search('Cdk5')[0]
activities = new_client.activity.filter(target_chembl_id=cdk5['target_chembl_id'])
activity_data = pd.DataFrame(activities)
activity_data.to_csv('ChEMBL data_raw.csv')
activity_data = activity_data[['canonical_smiles', 'standard_value']].dropna()
activity_data.rename(columns={'canonical_smiles': 'SMILES', 'standard_value': 'Activity'}, inplace=True)
activity_data['Activity'] = pd.to_numeric(activity_data['Activity'], errors='coerce')
activity_data.dropna(subset=['Activity'], inplace=True)
activity_data['Activity'] = (activity_data['Activity'] <= 1000).astype(int)

# Retrieve PubChem data
pubchem_data = []
compounds = get_compounds('Cdk5', 'name')
for compound in compounds:
    if compound.canonical_smiles:
        pubchem_data.append({'SMILES': compound.canonical_smiles, 'Activity': 1})

pubchem_df = pd.DataFrame(pubchem_data)

# Combine datasets
combined_data = pd.concat([activity_data, pubchem_df]).drop_duplicates(subset='SMILES')
combined_data.to_csv('combined_data.csv')
# Preprocess data
valid_smiles, refined_mols = refine_and_filter_data(combined_data)
combined_data = combined_data[combined_data['SMILES'].isin(valid_smiles)]
fingerprints = compute_fingerprints(refined_mols)
geo_chem_features = compute_geometrical_chemical_features(refined_mols)
fingerprint_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
combined_features = pd.concat([fingerprint_df, geo_chem_features], axis=1)
combined_features.fillna(0, inplace=True)

# Visualize class distribution before SMOTE
sns.countplot(x=combined_data['Activity'])
plt.title('Class Distribution Before SMOTE')
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, combined_data['Activity'], test_size=0.2, random_state=42
)

# Apply SMOTE to balance data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Visualize class distribution after SMOTE
sns.countplot(x=y_train_balanced)
plt.title('Class Distribution After SMOTE')
plt.show()

# Train and evaluate models
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [10, 20, 50, 100, 200], 'max_depth': [5,10, 20,30, None]}
rf_grid = GridSearchCV(rf, rf_params, cv=10, scoring='roc_auc')
rf_grid.fit(X_train_balanced, y_train_balanced)
rf_best = rf_grid.best_estimator_

svm = SVC(probability=True, random_state=42)
svm_params = {'C': [0.1, 1,5, 10,15], 'kernel': ['linear', 'rbf','polynomial']}
svm_grid = GridSearchCV(svm, svm_params, cv=10, scoring='roc_auc')
svm_grid.fit(X_train_balanced, y_train_balanced)
svm_best = svm_grid.best_estimator_

# Evaluate models
rf_preds = rf_best.predict(X_test)
rf_probs = rf_best.predict_proba(X_test)[:, 1]
svm_preds = svm_best.predict(X_test)
svm_probs = svm_best.predict_proba(X_test)[:, 1]

# Save processed data and models
combined_features.to_csv('processed_cdk5_data_combined_features.csv', index=False)
joblib.dump(rf_best, 'rf_model_with_features.pkl')
joblib.dump(svm_best, 'svm_model_with_features.pkl')

# Print metrics
print("Random Forest Metrics:")
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

print("\nSVM Metrics:")
print(classification_report(y_test, svm_preds))
print("ROC-AUC:", roc_auc_score(y_test, svm_probs))

# Plot ROC curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.2f})', color='blue')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {roc_auc_score(y_test, svm_probs):.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

###The next part is for testing the saved trained models and extract the 5 most proposed cdk5 inhibitors by the models average probabilities
import pandas as pd
import requests
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
import joblib

# Load the pre-trained models
rf_model = joblib.load("rf_model_with_features.pkl")
svm_model = joblib.load("svm_model_with_features.pkl")

# Query ChEMBL for CDK5-related bioactivity data
def fetch_cdk5_data():
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": "CHEMBL4015",  # ChEMBL ID for CDK5
        "activity_type": "IC50",
        "limit": 1000
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    data = response.json()["activities"]
    return pd.DataFrame(data)

# Fetch data from ChEMBL
data = fetch_cdk5_data()

# Preprocess and filter data
data = data.dropna(subset=["standard_value", "canonical_smiles", "molecule_chembl_id"])
data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
data = data[data["standard_value"] > 0]

# Group by molecule_chembl_id and compute average IC50
aggregated_data = data.groupby("molecule_chembl_id").agg({
    "standard_value": "mean",
    "canonical_smiles": "first"
}).reset_index()
aggregated_data.rename(columns={"standard_value": "average_ic50"}, inplace=True)

# Helper functions
salt_remover = SaltRemover()
neutralizer = rdMolStandardize.Uncharger()

def compute_fingerprints(mol_list):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fps = [np.array(morgan_gen.GetFingerprintAsNumPy(mol)) for mol in mol_list]
    return np.array(fps)

def compute_geometrical_chemical_features(mol_list):
    descriptor_names = [
        'MolWt', 'MolLogP', 'MolMR', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors', 'TPSA',
        'BalabanJ', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
        'Kappa1', 'Kappa2', 'Kappa3', 'FractionCSP3', 'HeavyAtomCount',
        'RingCount', 'NHOHCount', 'NOCount'
    ]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    features = [calc.CalcDescriptors(mol) for mol in mol_list]
    return pd.DataFrame(features, columns=descriptor_names)

# Preprocess SMILES data
valid_smiles = []
refined_mols = []
for smi in aggregated_data['canonical_smiles']:
    mol = refine_structure(smi)
    if mol:
        valid_smiles.append(smi)
        refined_mols.append(mol)

# Compute features
fingerprints = compute_fingerprints(refined_mols)
geo_chem_features = compute_geometrical_chemical_features(refined_mols)
fingerprint_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
combined_features = pd.concat([fingerprint_df, geo_chem_features], axis=1)

# Predict activity
rf_probs = rf_model.predict_proba(combined_features)[:, 1]
svm_probs = svm_model.predict_proba(combined_features)[:, 1]
aggregated_data['RF_Predicted_Activity'] = rf_probs
aggregated_data['SVM_Predicted_Activity'] = svm_probs
aggregated_data['Average_Activity'] = (rf_probs + svm_probs) / 2

# Select top 5 compounds
top_compounds = aggregated_data.nlargest(5, 'Average_Activity')

# Save results
top_compounds.to_csv('top_5_cdk5_inhibitors.csv', index=False)

print("Top 5 compounds have been saved to 'top_5_cdk5_inhibitors.csv'.")



