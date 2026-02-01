import torch
import torch.nn as nn

# A simple brain that predicts research success
class SuccessPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8), 
            nn.ReLU(),
            nn.Linear(8, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Simple function to use the model as a "Tool"
def get_experiment_prediction(compute, data):
    model = SuccessPredictor()
    # Loading dummy weights for explanation
    input_tensor = torch.tensor([[float(compute), float(data)]])
    prediction = model(input_tensor)
    return prediction.item()