from dataset import get_dataloaders
from model import build_model
from train import start_epochs

def main():
    train_loader, test_loader, classes = get_dataloaders(batch_size=64)
    model = build_model(num_classes=10)

    print("Training model...")
    start_epochs(model, train_loader, test_loader, epochs=30)

if __name__ == "__main__":
    main()