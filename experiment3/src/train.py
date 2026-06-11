import os

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(model, history, save_path):
    """保存模型权重与训练历史"""
    ensure_dir = os.path.dirname(save_path)
    if not os.path.exists(ensure_dir):
        os.makedirs(ensure_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_path)
    print(f"    [模型已保存] {save_path}")


def load_checkpoint(model, save_path, device='cpu'):
    """加载模型权重，返回 history（若存在）"""
    if not os.path.exists(save_path):
        return None
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    [模型已加载] {save_path}")
    return checkpoint.get('history')


def train_model(model, trainloader, epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    model.to(device)

    history = {'loss': [], 'acc': []}
    print(f"--- 开始训练 (设备: {device}) ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0;
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    return history


def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct = 0;
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total