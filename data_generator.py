import numpy as np
import torch


# torch.Tensor, [batch_size, no_nodes, 3]
def get_data_list(no_nodes, no_agents, batch_size):
    data = torch.rand((batch_size, no_nodes, 3))  # x, y, remaining_energy

    # depot
    data[:, 0, 0] = 0.5
    data[:, 0, 1] = 0.5
    data[:, 0, 2] = 1.0

    data[:, 1:, 2] = torch.rand((batch_size, no_nodes - 1)) * 0.8 + 0.2  # [0.2,1.0]

    return data

