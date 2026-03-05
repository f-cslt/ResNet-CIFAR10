import torch
import torch.nn as nn
import torch.optim as optim

def start_epochs(model, train_loader, test_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_model(model, test_loader, criterion,device)
        
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.3f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return model

def train_model(model, train_loader, criterion, optimizer, device):
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_loss += loss.item() * inputs.size(0)
        # Which class has the maximum value
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    
    avg_loss = total_loss / total
    train_acc = 100 * correct / total

    return avg_loss, train_acc

def test_model(model, test_loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total += labels.size(0)
            total_loss += loss.item() * inputs.size(0)
            # Which class has the maximum value
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        
        avg_loss = total_loss / total
        test_acc = 100 * correct / total

    
    return avg_loss, test_acc