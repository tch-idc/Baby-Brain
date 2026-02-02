import torch
import torch.nn as nn
import torch.optim as optim

# -------------------
# DATA 
# -------------------
# Learn: y = x1 + x2
X = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])

# -------------------
# MODEL 
# -------------------
class TinyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 1))  # w1, w2
        self.b = nn.Parameter(torch.randn(1))     # bias

    def forward(self, x):
        return x @ self.w + self.b


brain = TinyBrain()

print("Initial parameters:")
print("w:", brain.w)
print("b:", brain.b)

# -------------------
# TRAINING
# -------------------
optimizer = optim.SGD(brain.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
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
test_input = torch.tensor([[10.0, 20.0]])
prediction = brain(test_input)

print("\nFinal parameters:")
print("w:", brain.w)
print("b:", brain.b)

print(f"Prediction for {test_input.tolist()} =", prediction.item())
