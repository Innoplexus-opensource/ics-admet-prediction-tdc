"""
Author: Rohit Yadav
Organization: Innoplexus Consulting Services Pvt. Ltd., A Partex Company
"""

import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tdc.benchmark_group import admet_group

open('benchmark_log.txt', 'w').close()

def compute_morgan_fingerprints(smiles, radius=2, nBits=2024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    except:
        return np.nan
    return np.nan

def write_benchmark_result(benchmark_name, result):
    with open('benchmark_log.txt', 'a') as f:
        f.write(f"Results for {benchmark_name}: {result}\n")

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector, attention_weights

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, task):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.7)
        self.attention = AttentionLayer(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.task = task
        if task == 'binary':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x, attention_weights = self.attention(x.unsqueeze(1))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.activation(x)
        return x, attention_weights

benchmark_config = {
    'caco2_wang': ('regression', False),
    'bioavailability_ma': ('binary', True),
    'lipophilicity_astrazeneca': ('regression', False),
    'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', True),
    'pgp_broccatelli': ('binary', True),
    'bbb_martins': ('binary', True),
    'ppbr_az': ('regression', False),
    'vdss_lombardo': ('regression', False),
    'cyp2c9_veith': ('binary', True),
    'cyp2d6_veith': ('binary', True),
    'cyp3a4_veith': ('binary', True),
    'cyp2c9_substrate_carbonmangels': ('binary', True),
    'cyp2d6_substrate_carbonmangels': ('binary', True),
    'cyp3a4_substrate_carbonmangels': ('binary', True),
    'half_life_obach': ('regression', False),
    'clearance_hepatocyte_az': ('regression', False),
    'clearance_microsome_az': ('regression', False),
    'ld50_zhu': ('regression', False),
    'herg': ('binary', True),
    'ames': ('binary', True),
    'dili': ('binary', True),
    'hERG': ('binary', True),
    'hERG_Karim': ('binary', True),
    'AMES': ('binary', True),
    'DILI': ('binary', True),
    'Skin Reaction': ('binary', True),
    'LD50_Zhu': ('regression', False),
    'Carcinogens_Lagunin': ('binary', True),
    'ClinTox': ('binary', True)
}

logging.basicConfig(filename='benchmark_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')
os.makedirs('saved_models', exist_ok=True)
group = admet_group(path='data/')

for admet_benchmark in benchmark_config.keys():
    task, log_scale = benchmark_config[admet_benchmark]
    predictions_list = []
    for seed in [1, 2, 3, 4, 5]:
        benchmark = group.get(admet_benchmark)
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

        for df in [train, valid, test]:
            df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
            df['Fingerprint'] = df['Drug'].apply(lambda x: compute_morgan_fingerprints(x, nBits=1024))

        data = pd.concat([train, valid, test])
        valid_data = data.dropna(subset=['Fingerprint', 'Y'])
        X = np.stack(valid_data['Fingerprint'].values)
        y = valid_data['Y'].values

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        model = NeuralNetwork(input_dim=X_train.shape[1], task=task)
        
        if task == 'binary':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        for epoch in range(50):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    outputs, _ = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader)

            y_valid_pred, _ = model(X_valid_tensor)
            y_valid_pred = y_valid_pred.detach().numpy()

            if task == 'binary':
                y_valid_pred_class = (y_valid_pred > 0.5).astype(int)
                accuracy = accuracy_score(y_valid, y_valid_pred_class)
                recall = recall_score(y_valid, y_valid_pred_class)
                precision = precision_score(y_valid, y_valid_pred_class)
                f1 = f1_score(y_valid, y_valid_pred_class)
                auroc = roc_auc_score(y_valid, y_valid_pred)
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, "
                      f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, "
                      f"F1: {f1:.4f}, AUROC: {auroc:.4f}")
            else:
                mse = mean_squared_error(y_valid, y_valid_pred)
                r2 = r2_score(y_valid, y_valid_pred)
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, "
                      f"MSE: {mse:.4f}, R2: {r2:.4f}")

        test_data = test.dropna(subset=['Fingerprint', 'Y'])
        X_test = np.stack(test_data['Fingerprint'].values)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            y_pred, attention_weights = model(X_test_tensor)
            y_pred = y_pred.detach().numpy()
            attention_weights = attention_weights.detach().numpy()

        y_true_test = test_data['Y'].values

        if task == 'binary':
            y_pred_class = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_true_test, y_pred_class)
            recall = recall_score(y_true_test, y_pred_class)
            precision = precision_score(y_true_test, y_pred_class)
            f1 = f1_score(y_true_test, y_pred_class)
            auroc = roc_auc_score(y_true_test, y_pred)
            print(f"Seed {seed}: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
            logging.info(f"Seed {seed}: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
        else:
            mse = mean_squared_error(y_true_test, y_pred)
            r2 = r2_score(y_true_test, y_pred)
            print(f"Seed {seed}: MSE: {mse:.4f}, R2: {r2:.4f}")
            logging.info(f"Seed {seed}: MSE: {mse:.4f}, R2: {r2:.4f}")

        predictions[name] = y_pred.flatten()
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    print(f"Results for {name}: {results}")
    logging.info(f"Results for {name}: {results}")
    write_benchmark_result(name, results)