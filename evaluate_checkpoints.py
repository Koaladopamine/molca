import os
import torch
from data_provider.stage1_dm import Stage1DM
from model.blip2_stage1 import Blip2Stage1  

def evaluate_checkpoint(checkpoint_path, data_loader):
    # Load model from checkpoint
    model = Blip2Stage1.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    validation_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            loss = model.validation_step(batch, batch_idx)
        validation_loss += loss.item()
    
    validation_loss /= len(data_loader)
    return validation_loss

def main():
    # Arguments for the data loader
    args = {
        'num_workers': 2,
        'batch_size': 32,
        'root': 'data/PubChem324kV2',
        'text_max_len': 128,
        'graph_aug': True,
        'devices': [0]  # Example device setting
    }

    # Initialize data module
    data_module = Stage1KVPLMDM(
        num_workers=args['num_workers'],
        batch_size=args['batch_size'],
        root=args['root'],
        text_max_len=args['text_max_len'],
        graph_aug=args['graph_aug'],
        args=args
    )

    # Get validation data loader
    validation_loader = data_module.val_dataloader()

    # Directory where checkpoints are saved
    checkpoint_dir = "all_checkpoints/stage1"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    best_checkpoint = None
    best_metric = float('inf')  # Assuming lower is better (e.g., validation loss)

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        validation_loss = evaluate_checkpoint(checkpoint_path, validation_loader)
        
        print(f"Checkpoint {checkpoint_path} has validation loss: {validation_loss}")
        
        if validation_loss < best_metric:
            best_metric = validation_loss
            best_checkpoint = checkpoint_path

    print(f"Best checkpoint: {best_checkpoint} with validation loss: {best_metric}")

if __name__ == "__main__":
    main()
