import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np

from policy import Policy, action_sample, get_reward
from data_generator import get_data_list
from gapn_GNN import GNNPointer

log_file_path = "loss.txt"
dead_penalty = 10.


def train(batch_size, no_nodes, policy_net, l_r, no_agent, iterations, device, model):
    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    for itr in range(iterations):
        # prepare training data
        data = get_data_list(no_nodes, no_agent, batch_size)
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi, sample action and calculate log probabilities
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=batch_size)
        action, log_prob = action_sample(pi)

        # get reward and calculate loss using s-batch reinforce
        reward = get_reward(action, data, no_agent, dead_penalty, model)  # reward: tensor [batch, 1]
        mean_reward = sum(reward) / batch_size
        loss = torch.mul(torch.tensor(reward, device=device) - mean_reward, log_prob.sum(dim=1)).sum()

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print('\nIteration:', itr)
        print(format(mean_reward, '.4f'))

        # write file
        file_loss = format(mean_reward, '.4f')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Iteration {itr + 1}, Loss: {file_loss}\n')

        if (itr+1) % 100 == 0:
            torch.save(policy_net.state_dict(), './saved_model/{}_{}.pth'.format(str(no_nodes), str(no_agent)))


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    save_root = './GAPN_model/gnn_ptr50_10.0_lr1-5.pt'
    state = torch.load(save_root)
    model = GNNPointer(n_feature=3, n_hidden=128).cuda()
    model.load_state_dict(state['model'])

    n_agent = 5
    n_nodes = 50
    batch_size = 512
    lr = 1e-5
    iteration = 5000

    policy = Policy(in_chnl=3, hid_chnl=64, n_agent=n_agent, key_size_embd=128,
                    key_size_policy=128, val_size=128, clipping=10, dev=dev)

    train(batch_size, n_nodes, policy, lr, n_agent, iteration, dev, model)
