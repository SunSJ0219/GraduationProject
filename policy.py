import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

from gin import Net
from test_tsp import run_gapn
from test_tsp import run_gapn_val


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_reward(action, data, n_agent, dead_penalty, model):
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    depot = data[:, 0, :].tolist()

    # get clusters
    # sub_tours.shape = (batch, agent, node)
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])
        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)

    # Parallel Inference Strategy: by adding depots at the tail to make sub-tours equally aligned
    for i in range(data.shape[0]):
        filtered_subtours = [subtour for subtour in sub_tours[i] if len(subtour) > 1]
        depot = torch.tensor([0.5, 0.5, 1.0])
        max_len = max(len(lst) for lst in filtered_subtours)
        padded_subtours = pad_sequence(
            [torch.tensor(lst + [depot] * (max_len - len(lst))) for lst in filtered_subtours],
                                         batch_first=True)

        # masking is applied to ensure the additionally added points remain unaffected in GAPN
        mask = torch.zeros(padded_subtours.shape[0], max_len).cuda()
        for k in range(padded_subtours.shape[0]):
            for j in range(padded_subtours.shape[1]):
                if torch.equal(padded_subtours[k][j], depot):
                    mask[k][j] = -np.inf

        # calculate losses, GIN's loss is sub_graph's max loss
        time_cost, time_penalty = run_gapn(padded_subtours, mask, model)
        cost = time_cost + dead_penalty * time_penalty
        max_cost, _ = torch.max(cost, dim=0)
        max_cost = float(max_cost.to('cpu'))
        subtour_max_lengths[i] = max_cost

    return subtour_max_lengths


def get_reward_val(action, data, n_agent, dead_penalty, model):
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])
        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)

    xs = []
    ys = []
    time_costs = []
    time_penalties = []
    travel_costs = []

    for i in range(data.shape[0]):
        # print(i)
        filtered_subtours = [subtour for subtour in sub_tours[i] if len(subtour) > 1]
        # print(filtered_subtours)
        depot = torch.tensor([0.5, 0.5, 1.0])
        max_len = max(len(lst) for lst in filtered_subtours)
        padded_subtours = pad_sequence(
            [torch.tensor(lst + [depot] * (max_len - len(lst))) for lst in filtered_subtours],
            batch_first=True)
        # print(padded_subtours)
        mask = torch.zeros(padded_subtours.shape[0], max_len).cuda()
        for k in range(padded_subtours.shape[0]):
            for j in range(padded_subtours.shape[1]):
                if torch.equal(padded_subtours[k][j], depot):
                    mask[k][j] = -np.inf

        travel_cost, time_cost, time_penalty, ans_x, ans_y = run_gapn_val(padded_subtours, mask, model)
        travel_costs.append(travel_cost)
        time_costs.append(time_cost)
        time_penalties.append(time_penalty)
        xs.append(ans_x)
        ys.append(ans_y)
        cost = time_cost + dead_penalty * time_penalty
        # print(time_cost, time_penalty)
        max_cost, _ = torch.max(cost, dim=0)
        max_cost = float(max_cost.to('cpu'))
        # print(max_cost)
        subtour_max_lengths[i] = max_cost

    travel_cost_list = [value.tolist() for value in travel_costs]
    time_cost_list = [value.tolist() for value in time_costs]
    time_penalty_list = [value.tolist() for value in time_penalties]

    return subtour_max_lengths, travel_cost_list, time_cost_list, time_penalty_list, xs, ys

