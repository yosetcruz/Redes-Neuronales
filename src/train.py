import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
from sklearn.metrics import accuracy_score
from model import HousePurchaseMLP
from utils import load_and_preprocess, train_val_test_split

# Inicializar WandB
wandb.init(project="house-purchase-prediction", config={
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "optimizer": "Adam",
})
config = wandb.config

# Cargar datos (AJUSTA LA RUTA)
X, y, scaler = load_and_preprocess("../data/global_house_purchase_dataset.csv")
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

# Modelo
model = HousePurchaseMLP(input_dim=X_train.shape[1],
                         hidden_dims=config.hidden_dims,
                         dropout=config.dropout)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Entrenamiento
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    train_preds, train_true = [], []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        
        preds = (outputs > 0.5).float()
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(y_batch.cpu().numpy())
    
    train_loss /= len(train_loader.dataset)
    train_acc = accuracy_score(train_true, train_preds)
    
    # Validación
    model.eval()
    val_loss = 0.0
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(y_batch.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_true, val_preds)
    
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    })
    
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

# Evaluación final en test
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = (outputs > 0.5).float()
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(y_batch.cpu().numpy())
test_acc = accuracy_score(test_true, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
wandb.log({"test_acc": test_acc})

wandb.finish()

