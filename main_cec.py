from time import time
import numpy as np
from utils import visualize

import casadi as ca

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# This function implements receding-horizon certainty equivalent control
def cec_controller(cur_state, ref_state, obstacles, time_step, cur_iter):
    # V^*(\tau, e) \approx \min_{u_\tau, ..., u_{\tau + T - 1}}  \sum_{t=\tau}^{\tau+T-1} \gamma^{t-\tau}(\Tilde{p}_tQ\Tilde{p}_t + q(1 - \cos(\Tilde{\theta}_t))^2 + u_t^TRu_t)
    # such that 1. e_{t+1} = g(t, e_t, u_t, 0), t = \tau, ..., \tau + T - 1
    # 2. u_t \in U
    # 3. \Tilde{p}_t + r_t \in F

    # Q 2x2 is a symmetric positive definite matrix, the stage cost for deviating from the reference position trajectory
    Q = ca.DM([[10, 0], [0, 10]])
    # R 2x2 is a symmetric positive definite matrix, the stage cost for using excessive control effort
    R = ca.DM([[1, 0], [0, 0.1]])
    # q > 0 is the stage cost for deviating from the reference orientation trajectory
    q = 20
    # gamma \in (0, 1) is a scalar defining the discount factor
    gamma = 0.8



    ob_xy1 = obstacles[0][ : 2]
    ob_xy2 = obstacles[1][ : 2]
    ob_r1 = obstacles[0][2]
    ob_r2 = obstacles[1][2]


    # tilde_pt = pt - rt, tilde_theta_t = theta_t - alpha_t
    # convert cur_state to numpy array
    cur_state = np.array(cur_state)
    pt = ca.DM([cur_state[0], cur_state[1]])
    rt = ca.DM([ref_state[0], ref_state[1]])
    tilde_pt = pt - rt
    tilde_theta_t = cur_state[2] - ref_state[2]
    print("ref state", ref_state)
    

    
    T = 4
    opti = ca.Opti()
    u_t = opti.variable(T, 2)
    cost = 0

        
    # u_t = [v_t, w_t], v_t is the linear velocity, w_t is the angular velocity
    # v_t = sqrt((x_t - x_ref)^2 + (y_t - y_ref)^2) / time_step
    # w_t = (theta_ref - theta_t) / time_step
    # we want to optimize u_t, u_{t+1}, ..., u_{t+T-1}


    for t in range(T):


        # constraint: e_{t+1} = e_t + G(e_t)u_t
        # G(e_t) = [time_step * cos(tilde_theta_t), 0; time_step * sin(tilde_theta_t), 0; 0, time_step]
        # Ge_t = ca.DM([[time_step * ca.cos(cur_state[2]), 0], [time_step * ca.sin(cur_state[2]), 0], [0, time_step]])
        # et = ca.vertcat(tilde_pt, tilde_theta_t)
        # add e_t_plus_1 = e_t + Ge_t @ u_t to the list of constraints


        cur_state = car_next_state(time_step, cur_state, u_t[t,:])
        ref_state_new = lissajous(cur_iter + t + 1)
        
        ref_state = ref_state_new

        # update the tilde_pt and tilde_theta_t
        pt = ca.vertcat(cur_state[0], cur_state[1])
        rt = ca.vertcat(ref_state[0], ref_state[1])
        tilde_pt = pt - rt
        tilde_theta_t= cur_state[2] - ref_state[2]

        cost += gamma**(t) * (tilde_pt.T @ Q @ tilde_pt + q * (1 - ca.cos(tilde_theta_t))**2 + u_t[t,:] @ R @ u_t[t,:].T)


        opti.subject_to(v_min <= u_t[t, 0])
        opti.subject_to(u_t[t, 0] <= v_max)
        opti.subject_to(w_min <= u_t[t, 1])
        opti.subject_to(u_t[t, 1] <= w_max)
        opti.subject_to(-3 <= pt[0])
        opti.subject_to(pt[0] <= 3)
        opti.subject_to(-3 <= pt[1])
        opti.subject_to(pt[1] <= 3)
        opti.subject_to((pt - ob_xy1).T @ (pt - ob_xy1) > (ob_r1)**2)
        opti.subject_to((pt - ob_xy2).T @ (pt - ob_xy2) > (ob_r2)**2)




    # add the cost function
    opti.minimize(cost)
    # solve the nonlinear optimization problem
    opti.solver('ipopt', {'ipopt.print_level': 0})
    sol = opti.solve()

    # get the optimal control
    u_opt = sol.value(u_t)
    minimal_cost = sol.value(cost)
    # print("u_opt", u_opt)
    # print("minimal cost", minimal_cost)

    return u_opt[0]



# This function implement the car dynamics
def car_next_state(time_step, cur_state, control):
    theta = cur_state[2]
    # rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    control = control.T
    # rot_3d_z = ca.DM([[ca.cos(theta), 0], [ca.sin(theta), 0], [0, 1]])
    # f = rot_3d_z @ control
    f0 = ca.cos(theta) * control[0]
    f1 = ca.sin(theta) * control[0]
    f2 = control[1]
    f = ca.vertcat(f0, f1, f2)
    result = cur_state + time_step*f
    return result

def car_next_state_numpy(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()
    
if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)


        ################################################################
        # Generate control input
        # control = simple_controller(cur_state, cur_ref)
        control = cec_controller(cur_state, cur_ref, obstacles, time_step, cur_iter)

        ################################################################

        # Apply control input
        next_state = car_next_state_numpy(time_step, cur_state, control, noise=False)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi ) - np.pi
        print("next_state", next_state)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        print("error", error)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=False)

