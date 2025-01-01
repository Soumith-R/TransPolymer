# test_train.py
import torch
from models.transformer.transformerdecoder import PolymerDecoder
from data.utils.preprocessing import PolymerDataProcessor
from data.utils.data_augmentation import PolymerDataAugmentor
import pandas as pd

def test_train_model():
    # Load a small sample dataset for testing
    sample_data = {
        'Polymer SMILES': ['CC(=O)OC1=CC=CC=C1C(=C)C(=O)O'],  # Example SMILES
        'Solvent SMILES': ['C(C)C'],
        'MW (Da)': [100.0],
        'PDI': [1.0],
        'Φ': [0.5],
        'P (mPa)': [100.0],
        'CP (°C)': [50.0],
        '1-Phase': ['positive']
    }
    test_dataframe = pd.DataFrame(sample_data)

    # Initialize data processor
    data_processor = PolymerDataProcessor()
    features, targets = data_processor.prepare_dataset(test_dataframe)

    # Create a small dataset for testing
    test_data = PolymerDataset(features, targets)

    # Initialize model
    model = PolymerDecoder(vocab_size=len(data_processor.vocab))

    # Dummy optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Test training loop
    model.train()
    for batch in DataLoader(test_data, batch_size=1, shuffle=True):
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['targets'])
        loss.backward()
        optimizer.step()
        print(f'Test Loss: {loss.item():.4f}')

if __name__ == '__main__':
    test_train_model()# test_train.py
