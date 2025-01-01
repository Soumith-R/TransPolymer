import torch
import pandas as pd
import logging
import os
from typing import Dict, Union, List
import psutil
import warnings
from models.transformer.transformerdecoder import PolymerDecoder
from data.utils.preprocessing import PolymerDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class PolymerPredictor:
    def __init__(self, model_path: str = 'checkpoints/model_best.pt'):
        self.model_path = model_path
        self.device = torch.device('cpu')
        self.data_processor = None
        self.model = None
        
    def _load_model(self):
        """Load model with memory optimization for CPU"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Load checkpoint with explicit weights_only=True
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=True  # Safe loading
            )
            
            # Get model configuration
            vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
            
            # Initialize model
            self.model = PolymerDecoder(
                vocab_size=vocab_size,
                d_model=768,
                n_layers=12,
                n_heads=12,
                d_ff=3072,
                dropout=0.1
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Clear memory
            del checkpoint
            torch.cuda.empty_cache()
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    @torch.inference_mode()
    def predict_single(self, smile_string: str) -> Dict[str, float]:
        """Make prediction for a single SMILE string"""
        if not isinstance(smile_string, str) or not smile_string.strip():
            raise ValueError("Invalid SMILE string")

        try:
            # Lazy loading
            if self.model is None:
                self._load_model()
            if self.data_processor is None:
                self.data_processor = PolymerDataProcessor()

            # Create input features with all required columns
            input_df = pd.DataFrame({
                'Polymer SMILES': [smile_string],
                'Solvent SMILES': [''],
                'MW (Da)': [1000.0],
                'PDI': [1.0],
                'Φ': [0.0],
                'P (mPa)': [1.0],
                'CP (°C)': [0.0],
                '1-Phase': ['positive']  # Added missing column
            })

            # Process input
            features, _ = self.data_processor.prepare_dataset(input_df)
            
            # Convert to tensors
            input_ids = torch.tensor(
                features['polymer_smiles'], 
                dtype=torch.long, 
                device=self.device
            )
            attention_mask = torch.tensor(
                features['attention_mask'], 
                dtype=torch.float, 
                device=self.device
            )
            
            # Create encoder output
            encoder_output = torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.model.d_model,
                device=self.device,
                dtype=torch.float32
            )

            # Get all predictions
            outputs = self.model(input_ids, encoder_output, attention_mask)
            
            # Get predictions with temperature range
            cp_pred = outputs['cloud_point'].squeeze().item()
            phase_prob = outputs['phase'].squeeze().item()
            
            # Add uncertainty metrics
            predictions = {
                'Cloud Point (°C)': cp_pred,
                'Cloud Point Range': f"{cp_pred-5:.1f} to {cp_pred+5:.1f}°C",
                'Phase Probability': phase_prob,
                'Phase': 'Two-Phase' if phase_prob < 0.5 else 'One-Phase',
                'Phase Confidence': f"{abs(phase_prob - 0.5) * 200:.1f}%",
                'Reliability Score': f"{min(100, (1 - abs(phase_prob - 0.5)) * 100):.1f}%",
                'SMILE': smile_string
            }

            # Clean memory
            del features, input_ids, attention_mask, encoder_output
            
            return predictions

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

def check_system_requirements():
    """Check if system meets minimum requirements"""
    min_memory_gb = 2.0
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    logging.info(f"Available memory: {available_memory:.2f}GB")
    
    if available_memory < min_memory_gb:
        logging.warning(f"Low memory! Minimum {min_memory_gb}GB recommended. Available: {available_memory:.2f}GB")
        return False
    return True

def main():
    try:
        # Check system requirements
        if not check_system_requirements():
            logging.error("System requirements not met")
            return

        # Example usage
        test_smile = "CC(C)(C)CC(CC(C)(C)C)C(=O)OCCN(CC)CC"
        predictor = PolymerPredictor()
        
        try:
            result = predictor.predict_single(test_smile)
            print("\nPrediction Results:")
            print(f"Input SMILE: {result['SMILE']}")
            print("-" * 50)
            print(f"Cloud Point: {result['Cloud Point (°C)']:.2f}°C")
            print(f"Expected Range: {result['Cloud Point Range']}")
            print(f"Phase: {result['Phase']}")
            print(f"Phase Probability: {result['Phase Probability']:.2f}")
            print(f"Prediction Confidence: {result['Phase Confidence']}")
            print(f"Reliability Score: {result['Reliability Score']}")
            print("\nNote: Low confidence (<10%) indicates uncertain prediction")
                    
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            
    finally:
        # Clean up
        if 'predictor' in locals():
            del predictor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
