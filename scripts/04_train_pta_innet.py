"""
Script 04: Train PTA-InNet on NVIDIA GPU / CPU with composite loss.
"""

import argparse
from pydwr_wg.pipelines.train import train_pta_innet

def main():
    parser = argparse.ArgumentParser(description="Train PTA-InNet model.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--samples", type=int, default=2, help="Number of SCIP dataset samples")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pta_innet_model.pt")
    args = parser.parse_args()

    print("--- Running Script 04: Train PTA-InNet ---", flush=True)
    train_pta_innet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_samples=args.samples,
        device=args.device,
        save_checkpoint_path=args.checkpoint,
    )

if __name__ == "__main__":
    main()
