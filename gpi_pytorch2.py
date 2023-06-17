import numpy as np
from tqdm import tqdm

import torch
from scipy.stats import multivariate_normal


"""
The following are the parameters for the GPI algorithm
"""

# linear velocity in control
v_min = 0
v_max = 1
# anglular veolicty in control
w_min = -1
w_max = 1
# square free space in the environment
f_min = -3
f_max = 3
theta_min = -np.pi
theta_max = np.pi
time_step = 0.5

# 0 to 1
n_v = 5
# -1 to 1
n_w = 5
# 31 # n_f = n_x = n_y
n_f = 15
# 31
n_theta = 30
obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
obstacles = torch.tensor(obstacles, dtype=torch.float32)

# this is the number of time iterations, not n_t seconds
n_t = 100

w = np.linspace(w_min, w_max, n_w)
v = np.linspace(v_min, v_max, n_v)
# each grid has size 0.2
x = np.linspace(f_min, f_max, n_f)
y = np.linspace(f_min, f_max, n_f)
theta = np.linspace(theta_min, theta_max, n_theta)
T = np.linspace(0, n_t, n_t)

# Q 2x2 is a symmetric positive definite matrix, the stage cost for deviating from the reference position trajectory
Q = torch.diag(torch.tensor([10, 10], dtype=torch.float32))
# R 2x2 is a symmetric positive definite matrix, the stage cost for using excessive control effort
R = torch.diag(torch.tensor([0.5, 0.5], dtype=torch.float32))
# q > 0 is the stage cost for deviating from the reference orientation trajectory
q = 10
# gamma \in (0, 1) is a scalar defining the discount factor
gamma = 0.6

sigma = torch.tensor([0.4**2, 0.4**2, 0.04**2], dtype=torch.float32)
covar = torch.diag(sigma)
inv_covar = torch.linalg.inv(covar)
det_covar = torch.linalg.det(covar)



"""
The following are the functions for the GPI algorithm
"""


# This function returns the reference point at time step k
def lissajous(k):
    """
    modified from the original function to vectorize the for loop
    """

    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / 50
    b = 3 * a
    T = np.round(2 * np.pi / (a * time_step))
    k = np.mod(k, T)
    delta = np.pi / 2
    t = k * time_step
    xref = xref_start + A * np.sin(a * t + delta)
    yref = yref_start + B * np.sin(b * t)
    v = [A * a * np.cos(a * t + delta), B * b * np.cos(b * t)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

def car_next_state(time_step, cur_state, control_list):
    theta = cur_state[2]
    theta = torch.fmod(cur_state[2] + 3*np.pi, 2 * np.pi) - np.pi
    rot_3d_z = torch.tensor([[torch.cos(theta), 0], [torch.sin(theta), 0], [0, 1]]).to('cuda')
    f = torch.matmul(rot_3d_z, control_list.T)
    
    return cur_state + time_step*f.T # (len(control_list), 3)



def get_error_state_cost_vectorized(error_state, u_list, Q, R, q):

    p_tilde = error_state[:2]
    theta_tilde = error_state[2]

    cost = torch.matmul(torch.matmul(p_tilde, Q), p_tilde) + q * (
        1 - torch.cos(theta_tilde)
    ) ** 2
    # modified from the original
    result = torch.sum(torch.matmul(u_list, R) * u_list, axis=1) + cost.item()
    return result


def get_next_error_state_vectorized(
    curr_error_state, u_list, curr_ref_state, next_ref_state, time_step
):
    """
    modified from the original function to vectorize the for loop
    """

    curr_state = curr_error_state + curr_ref_state
    theta = curr_state[2]
    rot_3d_z = torch.tensor([[torch.cos(theta), 0], [torch.sin(theta), 0], [0, 1]]).to('cuda')
    f = torch.matmul(rot_3d_z, u_list.T)
    next_state = curr_state + time_step*f.T
    new_error_states = next_state - next_ref_state

    new_error_states[:, 2] = torch.fmod(new_error_states[:, 2] + 3*np.pi, 2 * np.pi) - np.pi

    new_error_states = new_error_states.to('cuda')

    valid_error_state = check_error_state_valid_vectorized(
        new_error_states, obstacles, next_ref_state, f_min, f_max
    )

    return new_error_states, valid_error_state


def check_error_state_valid_vectorized(
    error_state_list, obstacles, ref_state, f_min, f_max
):
    """
    modified from the original function to vectorize the for loop
    """

    ref_xy = ref_state[:2]
    tilde_xy = error_state_list[:, :2]
    position = ref_xy + tilde_xy
    result_list = torch.ones(len(error_state_list), dtype=torch.bool)
    result_list = result_list.to('cuda')
    result_list &= position[:, 0] >= f_min
    result_list &= position[:, 0] <= f_max
    result_list &= position[:, 1] >= f_min
    result_list &= position[:, 1] <= f_max
    for obstacle in obstacles:
        result_list &= torch.norm(position - obstacle[:2]) >= obstacle[2]
    return result_list


def get_reference_trajectory(t):
    return [lissajous(itr) for itr in range(t)]



def get_transition_prob_vectorized_sd(mean_e_list, other_e, ref_state = None, valid_state = None):
    """
    modified from the original function to vectorize the for loop
    mean_e_list: len(u_list) x 3
    other_e: sth x 3
    """
    # now treat other_e passed in as state instead of error_state
    # treat mean_e_list also has state instead of error_state

    assert mean_e_list.shape[1] == 3
    front = (2 * np.pi) ** (-3 / 2) * det_covar ** (-1 / 2)
    diff = other_e[:, np.newaxis, :] - mean_e_list
    temp = -0.5 * torch.einsum("ijk,kl,ijl->ij", diff, inv_covar, diff)
    prob = front * torch.exp(temp)

    # set the prob to 0 if the error state is not valid
    # current_state = other_e + ref_state
    # current_state = other_e
    # valid_state = check_error_state_valid_vectorized(current_state, obstacles, ref_state, f_min, f_max)
    
    prob[~valid_state] = 0
    prob[prob < 1e-5] = 0
    
    P = prob / (torch.sum(prob, axis=0, keepdims=True) + 1e-5)
    P = P.T

    return P




if __name__ == "__main__":
    V_error_state = np.zeros((n_t, n_f, n_f, n_theta))
    P_error_state = np.zeros((n_t, n_f, n_f, n_theta, 2))

    control_seq = np.zeros((n_t, 2))
    ref_traj_seq = get_reference_trajectory(n_t)

    # iterate through all possible next states (optimized)
    i2, j2, k2 = np.meshgrid(np.arange(n_f), np.arange(n_f), np.arange(n_theta))
    next_error_state_neighbor = np.column_stack(
        (x[i2.ravel()], y[j2.ravel()], theta[k2.ravel()])
    )
    other_e_list = next_error_state_neighbor.reshape(-1, 3)

    # do value iteration on the error state (optimized)
    u1, u2 = np.meshgrid(v, w)
    u = np.column_stack((u1.ravel(), u2.ravel()))
    u_list = u.reshape(-1, 2)

    # move ex_idx, ey_idx, etheta_idx out of the loop
    ex_idx = ((other_e_list[:, 0] + 3) / 6 * (n_f - 1)).astype(int).reshape(-1)
    ey_idx = ((other_e_list[:, 1] + 3) / 6 * (n_f - 1)).astype(int).reshape(-1)
    etheta_idx = (
        ((other_e_list[:, 2] + np.pi) / (2 * np.pi) * (n_theta - 1))
        .astype(int)
        .reshape(-1)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    u_list = torch.from_numpy(u_list).float().to(device)
    other_e_list = torch.from_numpy(other_e_list).float().to(device)
    ref_traj_seq = torch.tensor(ref_traj_seq).float().to(device)
    V_error_state = torch.from_numpy(V_error_state).float().to(device)
    P_error_state = torch.from_numpy(P_error_state).float().to(device)
    ex_idx = torch.from_numpy(ex_idx).long().to(device)
    ey_idx = torch.from_numpy(ey_idx).long().to(device)
    etheta_idx = torch.from_numpy(etheta_idx).long().to(device)
    Q = Q.to(device)
    R = R.to(device)
    obstacles = obstacles.to(device)
    inv_covar = inv_covar.to(device)
    det_covar = det_covar.to(device)
    
    # get transition_matrix, given the current state and control
    
    initialized = True
    if not initialized:
        transition_matrix = torch.empty((n_f, n_f, n_theta, len(u_list), 3))
        valid_state_matrix = torch.ones(len(other_e_list), dtype=torch.bool).to(device)
        transition_prob_matrix = np.empty((n_f, n_f, n_theta, len(u_list), len(other_e_list)))
        for obstacle in obstacles:
            valid_state_matrix &= torch.norm(other_e_list[:,:2] - obstacle[:2]) >= obstacle[2]
        valid_state_matrix = valid_state_matrix.cpu().numpy()
        pbar = tqdm(total=n_f*n_f*n_theta)
        for i in range(n_f):
            for j in range(n_f):
                for k in range(n_theta):
                    pbar.update(1)
                    cur_state = torch.tensor([x[i], y[j], theta[k]], dtype=torch.float32).to(device)
                    next_states = car_next_state(time_step, cur_state, u_list)
                    transition_matrix[i, j, k, :] = next_states
                    # transition_prob_matrix[i, j, k, :] = get_transition_prob_vectorized_sd(
                    #     next_states, other_e_list, valid_state = valid_state_matrix)

                    count = 0
                    for state in next_states:
                        transition_prob_matrix[i, j, k, count, :] = multivariate_normal.pdf(
                            other_e_list.cpu().numpy(), mean=state.cpu().numpy(), cov=[0.4, 0.4, 0.04]
                        )
                        transition_prob_matrix[i, j, k, count, :] /= np.sum(transition_prob_matrix[i, j, k, count, :]) + 1e-6
                        # set invalid states to 0
                        transition_prob_matrix[i, j, k, count, ~valid_state_matrix] = 0
                        count += 1
        pbar.close()

        transition_matrix = transition_matrix.cpu().numpy()
        transition_prob_matrix = transition_prob_matrix
        np.save("transition_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step), transition_matrix)
        np.save("valid_state_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step), valid_state_matrix)
        np.save("transition_prob_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step), transition_prob_matrix)

    transition_matrix = torch.from_numpy(np.load("transition_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step))).float().to(device)
    valid_state_matrix = torch.from_numpy(np.load("valid_state_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step))).bool().to(device)
    transition_prob_matrix = torch.from_numpy(np.load("transition_prob_matrix_{0}_{1}_{2}.npy".format(n_f, n_theta, time_step))).float().to(device)
    
    

    total_itr = int(15)
    pbar = tqdm(total=(total_itr+1) * (n_t - 1) * n_f * n_f * n_theta)
    for it in range(total_itr):
        V_error_state_this_itr = torch.clone(V_error_state)
        for t in range(n_t - 1):
            for i in range(n_f):
                for j in range(n_f):
                    for k in range(n_theta): 
                        pbar.update(1) 
                    
                        # curr_error_state = torch.tensor([x[i], y[j], theta[k]], dtype=torch.float32).to(device)
                        # curr_error_state_loss = get_error_state_cost_vectorized(curr_error_state, u_list, Q, R, q)

                        # (
                        #     new_error_states,
                        #     valid_states,
                        # ) = get_next_error_state_vectorized(
                        #     curr_error_state,
                        #     u_list,
                        #     ref_traj_seq[t],
                        #     ref_traj_seq[t + 1],
                        #     time_step,
                        # )
                        

                        curr_state = torch.tensor([x[i], y[j], theta[k]], dtype=torch.float32).to(device)
                        curr_error_state = curr_state - ref_traj_seq[t]
                        curr_error_state_loss = get_error_state_cost_vectorized(curr_error_state, u_list, Q, R, q)
                        
                        x_idx = i
                        y_idx = j
                        theta_idx = k
                        
                        # x_idx = ((curr_state[0] + 3) / 6 * (n_f - 1)).long().reshape(-1)
                        # y_idx = ((curr_state[1] + 3) / 6 * (n_f - 1)).long().reshape(-1)
                        # theta_idx = (
                        #     ((curr_state[2] + np.pi) / (2 * np.pi) * (n_theta - 1))
                        #     .long().reshape(-1)
                        # )
                        # if (x_idx.item() < 0) or (x_idx.item() >= n_f) or (y_idx.item() < 0) or (y_idx.item() >= n_f) or (theta_idx.item() < 0) or (theta_idx.item() >= n_theta):
                        #     # V_error_state_this_itr[t, i, j, k] = 1e10
                        #     continue

                        new_states = transition_matrix[x_idx, y_idx, theta_idx, :].reshape(-1, 3)
                        # new_error_states = (new_state - ref_traj_seq[t + 1]).reshape(-1, 3).to(device)
                        # new_error_states[:, 2] = torch.fmod(new_error_states[:, 2] + 3*np.pi, 2 * np.pi) - np.pi
                        # valid_states = valid_state_matrix[x_idx, y_idx, theta_idx, :].reshape(-1).to(device)
                        valid_states = valid_state_matrix
                        
                        prob = transition_prob_matrix[x_idx, y_idx, theta_idx, :]
                        assert prob.size() == (len(u_list), len(other_e_list))
                        assert prob.any() >= 0 and prob.any() <= 1

                        t_broadcast = torch.ones((len(other_e_list)), dtype=torch.long) * (t + 1)
                        expected_value = torch.einsum(
                            "ij, j->i",
                            prob,
                            V_error_state_this_itr[t_broadcast, ex_idx, ey_idx, etheta_idx],
                        )


                        loss_total = curr_error_state_loss + gamma * expected_value

                        min_cost, control_index = torch.min(loss_total, dim=0)
                        u = u_list[control_index]
                        if min_cost < 1e10:
                            V_error_state[t, i, j, k] = min_cost

        print("V max change: ", torch.max(torch.abs(V_error_state - V_error_state_this_itr)))
        if (torch.max(torch.abs(V_error_state - V_error_state_this_itr)) < 1):
            break

    V_error_state = V_error_state.detach().cpu().numpy()
    np.save("V_error_state_sd_nt100_nf15.npy", V_error_state)

    for t in range(n_t - 1):
        for i in range(n_f):
            for j in range(n_f):
                for k in range(n_theta):
                    pbar.update(1)
                    curr_state = torch.tensor([x[i], y[j], theta[k]], dtype=torch.float32).to(device)
                    curr_error_state = curr_state - ref_traj_seq[t]
                    curr_error_state_loss = get_error_state_cost_vectorized(curr_error_state, u_list, Q, R, q)
                    
                    x_idx = i
                    y_idx = j
                    theta_idx = k

                    new_states = transition_matrix[x_idx, y_idx, theta_idx, :].reshape(-1, 3)
                    valid_states = valid_state_matrix
                    prob = transition_prob_matrix[x_idx, y_idx, theta_idx, :]
                    assert prob.size() == (len(u_list), len(other_e_list))
                    assert prob.any() >= 0 and prob.any() <= 1
                    t_broadcast = torch.ones((len(other_e_list)), dtype=torch.long) * (t + 1)
                    expected_value = torch.einsum(
                        "ij, j->i",
                        prob,
                        V_error_state_this_itr[t_broadcast, ex_idx, ey_idx, etheta_idx],
                    )
                    loss_total = curr_error_state_loss + gamma * expected_value
                    min_cost, control_index = torch.min(loss_total, dim=0)
                    u = u_list[control_index]
                    P_error_state[t, i, j, k] = u

                    
                        

    P_error_state = P_error_state.detach().cpu().numpy()
    np.save("P_error_state_sd_nt100_nf15.npy", P_error_state)
