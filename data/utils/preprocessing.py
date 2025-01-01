# polymer_property_predictor/data/utils/preprocessing.py

import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from data.utils.data_augmentation import PolymerDataAugmentor

class PolymerDataProcessor:
    def __init__(self, max_length: int = 150):
        """
        Initialize the data processor with maximum sequence length.
        
        Args:
            max_length: Maximum length for SMILES strings padding
        """
        self.max_length = max_length
        self.atom_vocab = set()
        self.special_tokens = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]'
        }
        self.normalization_params = {}  # Initialize normalization parameters
        self.augmentor = PolymerDataAugmentor()
        self.column_mappings = {
            'MW(Da)': 'MW (Da)',
            'P(mPa)': 'P (mPa)',
            'CP(°C)': 'CP (°C)',
            'Solvent Smiles': 'Solvent SMILES'
        }
        
    def process_smiles(self, smiles: str) -> str:
        """
        Process SMILES string to ensure consistency.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Processed SMILES string
        """
        # Handle polymer specific tokens
        smiles = smiles.replace('*', '[*]')  # Handle polymer end groups
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol, canonical=True)
    
    def create_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
        """
        Create vocabulary from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping tokens to indices
        """
        # Add special tokens
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Process all SMILES and add unique tokens
        for smiles in smiles_list:
            processed = self.process_smiles(smiles)
            if processed:
                # Split into atoms and bonds
                tokens = []
                current_token = ''
                for char in processed:
                    if char.isupper():
                        if current_token:
                            tokens.append(current_token)
                        current_token = char
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
                    
                # Add to vocabulary
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
        
        return vocab
    
    def normalize_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical properties.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized properties
        """
        props = ['MW (Da)', 'PDI', 'Φ', 'w', 'P (mPa)', 'CP (°C)']
        normalized_df = df.copy()
        
        for prop in props:
            if prop in df.columns:
                # Handle potential non-numeric values
                numeric_mask = pd.to_numeric(df[prop], errors='coerce').notna()
                values = df.loc[numeric_mask, prop].astype(float)
                
                # Add safety check for log transformation
                if prop == 'MW (Da)':
                    values = values.replace(0, np.nan)  # Replace zeros with NaN
                    values = values[values > 0]  # Filter out negative values
                    values = np.log10(values)
                
                # Normalize to [0, 1] range
                min_val = values.min()
                max_val = values.max()
                
                # Convert to float64 before normalization
                normalized_df[prop] = normalized_df[prop].astype(float)
                normalized_df.loc[numeric_mask, prop] = (values - min_val) / (max_val - min_val)
                
                # Store normalization parameters
                self.normalization_params[prop] = {
                    'min': min_val,
                    'max': max_val,
                    'log_transform': prop == 'MW (Da)'
                }
        
        return normalized_df
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare dataset for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of input features and target values
        """
        # Create a copy of the dataframe and standardize column names
        df = df.copy()
        for old_name, new_name in self.column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Add feature scaling
        self.scaler = StandardScaler()
        numerical_features = ['MW (Da)', 'PDI', 'P (mPa)']
        
        # Ensure numerical features exist and handle missing ones
        for feature in numerical_features:
            if feature not in df.columns:
                orig_feature = next((k for k, v in self.column_mappings.items() if v == feature), feature)
                if orig_feature in df.columns:
                    df[feature] = df[orig_feature]
                else:
                    df[feature] = 0.0  # Default value
                    print(f"Warning: Missing feature {feature}, using default value")

        # Convert numerical columns to float and handle any non-numeric values
        for feature in numerical_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)

        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Initialize normalization parameters dictionary
        self.normalization_params = {}
        
        # Normalize numerical properties
        normalized_df = self.normalize_properties(df)
        
        # Define and handle required columns
        required_columns = {
            'Solvent SMILES': '',
            'MW (Da)': 0,
            'PDI': 0,
            'Φ': 0,
            'w': 0,
            'P (mPa)': 0,
            'CP (°C)': 0
        }

        for column, default_value in required_columns.items():
            if column not in df.columns:
                orig_column = next((k for k, v in self.column_mappings.items() if v == column), column)
                if orig_column in df.columns:
                    df[column] = df[orig_column]
                else:
                    df[column] = default_value
        
        # Create vocabulary from both polymer and solvent SMILES
        all_smiles = pd.concat([
            df['Polymer SMILES'],
            df[self.column_mappings.get('Solvent Smiles', 'Solvent SMILES')]
        ]).unique()
        self.vocab = self.create_vocabulary(all_smiles)
        
        # Prepare input features with correct column names
        polymer_features = self.tokenize_and_pad(df['Polymer SMILES'].values)
        solvent_features = self.tokenize_and_pad(df[self.column_mappings.get('Solvent Smiles', 'Solvent SMILES')].values)
        
        features = {
            'polymer_smiles': polymer_features['input_ids'],
            'attention_mask': polymer_features['attention_mask'],
            'solvent_smiles': solvent_features['input_ids'],
            'mw': normalized_df['MW (Da)'].values,
            'pdi': normalized_df['PDI'].values,
            'phi': normalized_df['Φ'].values,
            'pressure': normalized_df['P (mPa)'].values
        }
        
        # Prepare target values with correct column names
        targets = {
            'cloud_point': normalized_df[self.column_mappings.get('CP(°C)', 'CP (°C)')].values,
            'phase': (df['1-Phase'] == 'positive').astype(int).values
        }
        
        return features, targets
    
    def tokenize_and_pad(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize and pad SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary containing token indices and attention masks
        """
        tokenized = np.zeros((len(smiles_list), self.max_length), dtype=int)
        attention_masks = np.zeros((len(smiles_list), self.max_length), dtype=int)
        pad_idx = self.vocab[self.special_tokens['PAD']]
        
        for i, smiles in enumerate(smiles_list):
            processed = self.process_smiles(smiles)
            if processed:
                tokens = []
                current_token = ''
                for char in processed:
                    if char.isupper():
                        if current_token:
                            tokens.append(current_token)
                        current_token = char
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
                
                # Convert tokens to indices
                token_indices = [self.vocab.get(token, self.vocab[self.special_tokens['UNK']]) 
                               for token in tokens]
                
                # Add CLS and SEP tokens
                token_indices = [self.vocab[self.special_tokens['CLS']]] + token_indices + [self.vocab[self.special_tokens['SEP']]]
                
                # Pad or truncate
                if len(token_indices) > self.max_length:
                    tokenized[i] = token_indices[:self.max_length]
                    attention_masks[i] = [1] * self.max_length
                else:
                    tokenized[i, :len(token_indices)] = token_indices
                    attention_masks[i, :len(token_indices)] = 1
                    
        return {
            'input_ids': tokenized,
            'attention_mask': attention_masks
        }
    
    def augment_smiles(self, smiles: str, n_augment: int = 5) -> List[str]:
        """
        Augment SMILES string with multiple techniques
        
        Args:
            smiles: Input SMILES string
            n_augment: Number of augmentations to generate
            
        Returns:
            List of augmented SMILES strings
        """
        augmented = []
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return [smiles]
            
        try:
            # Random SMILES generation
            for _ in range(n_augment):
                aug = Chem.MolToSmiles(mol, doRandom=True)
                if aug and aug not in augmented:
                    augmented.append(aug)
                    
            # Chemical substitutions
            subst_aug = self.augmentor.random_substitution(smiles)
            if subst_aug and subst_aug not in augmented:
                augmented.append(subst_aug)
                
        except Exception as e:
            print(f"Error during augmentation: {str(e)}")
            return [smiles]
            
        if not augmented:
            augmented.append(smiles)
            
        return augmented[:n_augment]