import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Xi-AI 范式验证 - 非常简化的测试")

DEVICE = "cpu"

class SimpleXIMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.rho = nn.Parameter(torch.tensor([0.55]))  # 单个零点实部

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

    def rh_loss(self):
        return (self.rho - 0.5).abs().mean()

def main():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_clusters_per_class=1, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

    print(f"数据集: {len(X_train)} train, {len(X_test)} test")

    # 测试 λ_RH = 0 和 λ_RH = 0.1
    for lambda_rh in [0.0, 0.01, 0.1]:
        model = SimpleXIMLP(X_train.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

        best_acc = 0
        epochs = 30
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            ce = F.cross_entropy(pred, y_train_t)
            rh = model.rh_loss()
            loss = ce + lambda_rh * rh * 5.0
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_acc = (test_pred.argmax(1) == y_test_t).float().mean().item()
            val_pred = model(X_train_t)
            train_acc = (val_pred.argmax(1) == y_train_t).float().mean().item()

        print(f"λ_RH={lambda_rh:.3f}: Train={train_acc:.3f}, Test={test_acc:.3f}, RH={model.rho.item():.3f}")

    print("\n✅ 验证完成！")

if __name__ == "__main__":
    main()
