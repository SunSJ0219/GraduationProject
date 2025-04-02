import numpy as np
import torch
from gapn_GNN import GNNPointer
charging_rate = 0.25
drain_rate = 5


def run_gapn(data, mask, model):
    baseline = 0

    batch_size = data.shape[0]
    no_nodes = data.shape[1]
    ddl = data[:, :, 2]
    X = torch.Tensor(data).cuda()
    ddl = torch.Tensor(ddl).cuda()

    time_penalty = torch.zeros(batch_size).cuda()
    total_time_cost = torch.zeros(batch_size).cuda()
    total_time_penalty = torch.zeros(batch_size).cuda()

    travel_length = torch.zeros(batch_size).cuda()

    cur_time = torch.zeros(batch_size).cuda()
    ser_time = torch.zeros(batch_size).cuda()
    x = X[:, 0, :]
    h = None
    c = None

    for k in range(no_nodes):
        output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
        idx = torch.argmax(output, dim=1)  # greedy baseline
        # ans.append(idx)
        y_cur = X[[i for i in range(batch_size)], idx.data].clone()
        # print(y_cur[0])
        if k == 0:
            y_ini = y_cur.clone()
        else:
            elementwise_equal = torch.eq(y_cur, y_ini)
            all_equal = torch.all(elementwise_equal, dim=1)

            travel_cost = torch.norm(y_cur[:, :2] - y_pre[:, :2], dim=1)
            cur_time += travel_cost
            deadline = ddl[[i for i in range(batch_size)], idx.data]
            deadline_exhausted = torch.lt(deadline * drain_rate, cur_time)
            max_charge = (torch.tensor(1.0) / charging_rate).cuda()
            service_time = torch.where(deadline_exhausted, max_charge,
                                       (1.0 - (y_cur[:, 2] * drain_rate - cur_time * drain_rate)) / charging_rate)
            service_time[all_equal] = 0.
            deadline_exhausted[all_equal] = False
            cur_time += service_time
            total_time_cost += travel_cost + service_time
            total_time_penalty += deadline_exhausted.float()
        y_pre = y_cur.clone()
        x = X[[i for i in range(batch_size)], idx.data]

        mask[[i for i in range(batch_size)], idx.data] += -np.inf
    total_time_cost += torch.norm(y_cur[:, :2] - y_ini[:, :2], dim=1)
    return total_time_cost, total_time_penalty


def run_gapn_val(data, mask, model):
    baseline = 0

    batch_size = data.shape[0]
    no_nodes = data.shape[1]
    ddl = data[:, :, 2]
    X = torch.Tensor(data).cuda()
    ddl = torch.Tensor(ddl).cuda()

    total_travel_cost = torch.zeros(batch_size).cuda()
    total_time_cost = torch.zeros(batch_size).cuda()
    total_time_penalty = torch.zeros(batch_size).cuda()

    travel_length = torch.zeros(batch_size).cuda()

    cur_time = torch.zeros(batch_size).cuda()
    ser_time = torch.zeros(batch_size).cuda()
    x = X[:, 0, :]
    h = None
    c = None

    ans_x = [[] for _ in range(batch_size)]
    ans_y = [[] for _ in range(batch_size)]

    for k in range(no_nodes):
        output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
        idx = torch.argmax(output, dim=1)  # greedy baseline
        # ans.append(idx)
        y_cur = X[[i for i in range(batch_size)], idx.data].clone()
        # print(y_cur[0])
        for i in range(batch_size):
            ans_x[i].append(y_cur[i][0].item())
            ans_y[i].append(y_cur[i][1].item())
        if k == 0:
            y_ini = y_cur.clone()
        else:
            elementwise_equal = torch.eq(y_cur, y_ini)
            all_equal = torch.all(elementwise_equal, dim=1)

            travel_cost = torch.norm(y_cur[:, :2] - y_pre[:, :2], dim=1)
            cur_time += travel_cost
            total_travel_cost += travel_cost
            deadline = ddl[[i for i in range(batch_size)], idx.data]
            deadline_exhausted = torch.lt(deadline * drain_rate, cur_time)
            max_charge = (torch.tensor(1.0) / charging_rate).cuda()
            service_time = torch.where(deadline_exhausted, max_charge,
                                       (1.0 - (y_cur[:, 2] * drain_rate - cur_time * drain_rate)) / charging_rate)
            service_time[all_equal] = 0.
            deadline_exhausted[all_equal] = False
            cur_time += service_time
            total_time_cost += travel_cost + service_time
            total_time_penalty += deadline_exhausted.float()
        y_pre = y_cur.clone()
        x = X[[i for i in range(batch_size)], idx.data]

        mask[[i for i in range(batch_size)], idx.data] += -np.inf
    for i in range(batch_size):
        ans_x[i].append(y_ini[i][0].item())
        ans_y[i].append(y_ini[i][1].item())
    total_time_cost += torch.norm(y_cur[:, :2] - y_ini[:, :2], dim=1)
    return total_travel_cost, total_time_cost, total_time_penalty, ans_x, ans_y


if __name__ == '__main__':
    np.random.seed(42)
    data = torch.tensor([[[0.5, 0.5, 0.], [0.4, 0.4, .2], [0.1, 0.1, .5]], [[0.5, 0.5, 0.], [0.4, 0.4, .2], [0.1, 0.1, .5]]])
    # data = get_data_list(3, 1, 1)

    save_root = './GAPN_model/gnn_ptr20_delta10.0_20_lr1-5.pt'
    # save_root ='./model/GPN_withDDL50_high.pt'
    state = torch.load(save_root)
    model = GNNPointer(n_feature=3, n_hidden=128).cuda()
    model.load_state_dict(state['model'])

    mask = torch.zeros(data.shape[0], data.shape[1]).cuda()
    total_time_cost, total_time_penalty = run_gapn(data, mask, model)
    print(total_time_cost, total_time_penalty)
    mask = torch.zeros(data.shape[0], data.shape[1]).cuda()
    total_time_cost1, total_time_penalty1, xs, ys = run_gapn_val(data, mask, model)
    print(total_time_cost1, total_time_penalty1, xs, ys)
