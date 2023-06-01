import numpy as np
import main_gpi

def get_cost(curr_state_list, u_list, ref_state_list, Q, R, q, gamma):
    cost = 0
    for t in range(len(curr_state_list)):
        p_tilde = curr_state_list[t][:2] - ref_state_list[t][:2]
        theta_tilde = curr_state_list[t][2] - ref_state_list[t][2]
        cost += gamma**(t) * (p_tilde.T @ Q @ p_tilde + q * (1 - np.cos(theta_tilde))**2 + u_list[t] @ R @ u_list[t].T)
    return cost
 


def get_reference_trajectory(t):
    traj_list = []
    for itr in range(t):
        ref_state = main_gpi.lissajous(itr)
        traj_list.append(ref_state)
    return traj_list

def get_robot_trajectory(t_start, t_end, cur_state, u_seq):
    robot_traj = []
    robot_traj.append(cur_state)
    for itr in range(t_start, t_end):
        cur_state = main_gpi.car_next_state(1, cur_state, u_seq[itr-t_start])
        robot_traj.append(cur_state)
    return robot_traj

# given e, return a probability of transition to other_e with mean at e and variance sigma
# and make sure all the probability of other_es sum to 1
def get_transition_prob(mean_e, other_e):
    sigma = np.array([0.04, 0.04, 0.004])
    covar = np.diag(sigma**2)
    prob = np.zeros(len(other_e))
    for i in range(len(other_e)):
        prob[i] = np.exp(-0.5 * (other_e[i] - mean_e).T @ np.linalg.inv(covar) @ (other_e[i] - mean_e))
    prob = prob / np.sum(prob)
    





if __name__ == '__main__':
    v_min = 0 # linear velocity in control
    v_max = 1
    w_min = -1 # anglular veolicty in control
    w_max = 1
    f_min = -3 # square free space in the environment
    f_max = 3
    theta_min = -np.pi
    theta_max = np.pi

    n_v = 31
    n_w = 31
    n_f = 31
    n_theta = 31
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])

    n_t = 100

    w = np.linspace(w_min, w_max, n_w)
    v = np.linspace(v_min, v_max, n_v)
    x = np.linspace(f_min, f_max, n_f)
    y = np.linspace(f_min, f_max, n_f)
    theta = np.linspace(theta_min, theta_max, n_theta)
    T = np.linspace(0, n_t, n_t+1)
    
    # Q 2x2 is a symmetric positive definite matrix, the stage cost for deviating from the reference position trajectory
    Q = np.array([[1, 0], [0, 1]])
    # R 2x2 is a symmetric positive definite matrix, the stage cost for using excessive control effort
    R = np.array([[0.1, 0], [0, 0.1]])
    # q > 0 is the stage cost for deviating from the reference orientation trajectory
    q = 1
    # gamma \in (0, 1) is a scalar defining the discount factor
    gamma = 0.9

    V_error_state = np.zeros((n_t, n_f, n_f, n_theta))
    time_step = 1
    control_seq = np.zeros((n_t, 2))
    ref_traj_seq = get_reference_trajectory(n_t)

    


    # do value iteration on the error state
    # V^*(\tau,e) = \min_\pi E[\sum_{t=\tau}^\infty \gamma^{t-\tau}(\Tilde{p}_t^TQ\Tilde{p}_t + q(1 - \cos(\Tilde{\theta}))^2 + u_t^TRu_t)|e_\tau = e]


    # for t in range(n_t):
    #     print("time step: ", t)
    #     for i in range(n_f):
    #         for j in range(n_f):
    #             for k in range(n_theta):
    #                 curr_state = np.array([x[i], y[j], theta[k]])
    #                 min_cost = np.inf
    #                 for u1 in v:
    #                     for u2 in w:
    #                         u = np.array([u1, u2])
    #                         cost = 0
                            
                            # get the next state




    # policy extraction

                            

