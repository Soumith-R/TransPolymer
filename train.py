import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models.transformer.transformerdecoder import PolymerDecoder
from data.utils.preprocessing import PolymerDataProcessor
import numpy as np
import torch.nn as nn
import os
import math
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR

def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Available columns:", df.columns.tolist())
        
        column_mapping = {
            'MW(Da)': 'MW (Da)',
            'P(mPa)': 'P (mPa)', 
            'CP(°C)': 'CP (°C)',
            'Solvent Smiles': 'Solvent SMILES'
        }
        
        df = df.rename(columns=column_mapping)
        
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
                print(f"Adding missing column {col} with default value")
                df[col] = default_val
                
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

class PolymerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features['polymer_smiles'])

    def __getitem__(self, idx):
        # Convert input_ids to long (integer) type and ensure other tensors are float
        return {
            'input_ids': torch.tensor(self.features['polymer_smiles'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.features['attention_mask'][idx], dtype=torch.float),
            'targets': torch.tensor(self.targets['cloud_point'][idx], dtype=torch.float)
        }

class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 
        self.huber = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        huber_loss = self.huber(pred, target)
        return mse_loss + self.alpha * l1_loss + self.beta * huber_loss

def get_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        factor = max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return factor
    
    return LambdaLR(optimizer, lr_lambda)

def train_model(train_data, val_data, model, optimizer, criterion, num_epochs=100, save_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    batch_size = 64
    accum_iter = 8
    
    # Update scheduler configuration using custom function
    total_steps = num_epochs * len(train_data) // batch_size
    warmup_steps = total_steps // 10
    
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-6
    )
    
    # Add exponential moving average
    ema = torch.optim.swa_utils.AveragedModel(model)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Training loop with gradient accumulation
            for i, batch in enumerate(DataLoader(train_data, batch_size=batch_size, shuffle=True)):
                # Convert tensors to correct dtypes
                input_ids = batch['input_ids'].long().to(device)
                attention_mask = batch['attention_mask'].float().to(device)
                targets = batch['targets'].float().to(device)
                
                optimizer.zero_grad()
                
                # Create encoder output with float dtype
                encoder_output = torch.zeros(
                    input_ids.size(0), 
                    input_ids.size(1), 
                    model.d_model,
                    dtype=torch.float32,
                    device=device
                )
                
                # Ensure model processes correct dtypes
                outputs = model(
                    input_ids=input_ids,  # Long tensor
                    encoder_output=encoder_output,  # Float tensor
                    attention_mask=attention_mask.float()  # Float tensor
                )
                
                loss = criterion(outputs['cloud_point'].squeeze(), targets)
                loss = loss / accum_iter
                scaler.scale(loss).backward()
                
                if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update_parameters(model)
                
                total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_data)
        
        # Validation with correct dtypes
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in DataLoader(val_data, batch_size=32):
                input_ids = batch['input_ids'].long().to(device)
                attention_mask = batch['attention_mask'].float().to(device)
                targets = batch['targets'].float().to(device)
                
                encoder_output = torch.zeros(
                    input_ids.size(0), 
                    input_ids.size(1),
                    model.d_model,
                    dtype=torch.float32,
                    device=device
                )
                
                outputs = model(input_ids, encoder_output, attention_mask)
                val_loss += criterion(outputs['cloud_point'].squeeze(), targets).item()
        
        avg_val_loss = val_loss / len(val_data)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'scaler': scaler.state_dict(),  # Save scaler state
            }
            torch.save(checkpoint, f'{save_dir}/model_best.pt')
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    try:
        save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and process data
        data_processor = PolymerDataProcessor()
        train_df = load_dataset('data/raw/Polymer6kDataset.xlsx')
        
        if train_df is None:
            raise ValueError("Failed to load training data")
            
        # Split into train/val
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Process datasets
        train_features, train_targets = data_processor.prepare_dataset(train_df)
        val_features, val_targets = data_processor.prepare_dataset(val_df)
        
        train_data = PolymerDataset(train_features, train_targets)
        val_data = PolymerDataset(val_features, val_targets)
        
        # Initialize with better hyperparameters
        model = PolymerDecoder(
            vocab_size=len(data_processor.vocab),
           d_model=768,      # Increased from 512
            n_layers=12,      # Increased from 8  
            n_heads=12,       # Increased from 8
            d_ff=3072,        # Increased from 2048
            dropout=0.1   # Increased from 0.1
        )
        
        # Use AdamW optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Use combined loss with L1 regularization
        criterion = EnhancedLoss()

        # Updated train call with validation data
        train_model(train_data, val_data, model, optimizer, criterion, num_epochs=100, save_dir=save_dir)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
        # Updated train call with validation data
            torch.cuda.empty_cache()
        raise
        train_model(train_data, val_data, model, optimizer, criterion, num_epochs=100, save_dir=save_dir)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise