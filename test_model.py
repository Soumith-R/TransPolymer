import torch
from torch.utils.data import DataLoader, Dataset
from models.transformer.transformerdecoder import PolymerDecoder
from data.utils.preprocessing import PolymerDataProcessor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_results.log'
)

def load_test_data(file_path):
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Available columns: {df.columns.tolist()}")
        
        # Map columns
        column_mapping = {
            'MW(Da)': 'MW (Da)',
            'P(mPa)': 'P (mPa)', 
            'CP(°C)': 'CP (°C)',
            'Solvent Smiles': 'Solvent SMILES'
        }
        df = df.rename(columns=column_mapping)
        
        # Add missing columns with defaults
        optional_columns = {
            'Solvent SMILES': '',
            'MW (Da)': 1.0,
            'PDI': 1.0,
            'Φ': 0.0,
            'P (mPa)': 1.0,
            'CP (°C)': 0.0
        }
        
        for col, default_val in optional_columns.items():
            if col not in df.columns:
                logging.info(f"Adding missing column {col} with default value")
                df[col] = default_val
        
        # Validate data
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"CP (°C) range: [{df['CP (°C)'].min():.2f}, {df['CP (°C)'].max():.2f}]")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

class TestDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        logging.info(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.features['polymer_smiles'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.features['polymer_smiles'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.features['attention_mask'][idx], dtype=torch.float),
            'targets': torch.tensor(self.targets['cloud_point'][idx], dtype=torch.float)
        }

def evaluate_model(predictions, targets, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Prediction vs True Values (R² = {r2:.3f})')
    plt.savefig(os.path.join(save_dir, 'predictions.png'))
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'True_Values': targets,
        'Predictions': predictions
    })
    results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # Log metrics
    logging.info(f"\nModel Evaluation:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")
    
    return {
        'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
        'statistics': {
            'predictions': {'mean': np.mean(predictions), 'std': np.std(predictions)},
            'targets': {'mean': np.mean(targets), 'std': np.std(targets)}
        }
    }

def test_saved_model(test_data_path, checkpoint_path, batch_size=16):
    try:
        # Load data
        test_df = load_test_data(test_data_path)
        data_processor = PolymerDataProcessor()
        features, targets = data_processor.prepare_dataset(test_df)
        
        # Load checkpoint first to get vocab size
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        
        # Initialize model with matching architecture
        model = PolymerDecoder(
            vocab_size=vocab_size,
            d_model=768,      
            n_layers=12,      
            n_heads=12,       
            d_ff=3072,        
            dropout=0.1
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("Model loaded successfully")
        
        # Test data
        test_dataset = TestDataset(features, targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Make predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['targets']
                
                encoder_output = torch.zeros(input_ids.size(0), input_ids.size(1), model.d_model)
                outputs = model(input_ids, encoder_output, attention_mask)
                predictions = outputs['cloud_point'].squeeze()
                
                all_predictions.extend(predictions.numpy())
                all_targets.extend(targets.numpy())
                
                # Clean up memory
                del outputs, predictions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Evaluate results
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        evaluation = evaluate_model(all_predictions, all_targets)
        
        return all_predictions, evaluation
        
    except Exception as e:
        logging.error(f"Error in testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_path = 'data/raw/Polymer6kDataset.xlsx'
    checkpoint_path = 'checkpoints/model_best.pt'
    os.makedirs('results', exist_ok=True)
    
    predictions, evaluation = test_saved_model(test_data_path, checkpoint_path, batch_size=16)