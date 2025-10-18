import torch
from torch import nn
from torchvision import models, datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
base_model.heads = nn.Linear(in_features=768, out_features=100)
base_model = base_model.to(device)

cifar100_dataset_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]))
cifar100_train_loader = torch.utils.data.DataLoader(cifar100_dataset_train, batch_size=8, shuffle=True, num_workers=4)

cifar100_dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]))
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_dataset_test, batch_size=8, shuffle=False, num_workers=4)


def compute_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(cifar100_test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(cifar100_train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        accuracy = compute_accuracy(model)
        accuracy_history.append(accuracy)
        print(f'Accuracy after epoch {epoch+1}: {accuracy:.2f}%')
    
    return model, loss_history, accuracy_history

def plot_metrics(loss_history, accuracy_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(range(1, len(loss_history) + 1), loss_history, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(range(1, len(accuracy_history) + 1), accuracy_history, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'training_metrics.png'")
    plt.show()

if __name__ == "__main__":
    trained_model, loss_history, accuracy_history = train_model(base_model, epochs=int(input("Enter number of epochs for training: ")))
    print(f'Final Accuracy: {compute_accuracy(trained_model):.2f}%')
    plot_metrics(loss_history, accuracy_history)
    torch.save(trained_model.state_dict(), "cifar100_model.pth")
    print("Model saved successfully.")