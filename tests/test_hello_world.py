import torch
import logging

def load_model_state(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError as e:
        logging.error("Error loading model state_dict: %s", e)
        logging.error("Ensure that the model architecture matches the checkpoint.")
        raise

def test_load_model():
    model = YourModelClass()  # Replace with your actual model class
    checkpoint_path = 'path/to/your/checkpoint.pth'  # Update with your checkpoint path
    load_model_state(model, checkpoint_path)

if __name__ == "__main__":
    test_load_model()