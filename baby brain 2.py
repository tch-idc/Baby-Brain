import torch
import torch.nn as nn
import torch.optim as optim

# -------------------
# DATA 
# -------------------
X = torch.randn(500, 10)   # 500 samples, 10 features
y = X.sum(dim=1, keepdim=True)  # target = sum of inputs

# -------------------
# MODEL 
# -------------------
class Brain1K(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


brain = Brain1K()

# Count parameters
total_params = sum(p.numel() for p in brain.parameters())
print("Total parameters:", total_params)

# -------------------
# TRAINING
# -------------------
optimizer = optim.Adam(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(300):
    pred = brain(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------
# TEST
# -------------------
test_input = torch.randn(1, 10)
prediction = brain(test_input)

print("\nTest input:", test_input)
print("Prediction:", prediction.item())
print("True sum:", test_input.sum().item())

