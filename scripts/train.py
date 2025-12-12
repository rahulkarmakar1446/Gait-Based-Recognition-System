# scripts/train.py
import argparse
from src.model import build_model, train
from src.preprocessing import create_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, help='Path to data')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=32)
    model = build_model()
    train(model, train_loader, val_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
