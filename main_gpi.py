from time import time
import numpy as np
from utils import visualize
import gpi_pytorch2
from itertools import product

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 50    # simulation time, in this way n_t = 100

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
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise=True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04  # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step * f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()

def new_gpi_controller(cur_state, cur_ref, cur_iter, P_error_state):
    error_state = cur_state - cur_ref
    ########## remember to configure the parameters ##########
    n_f = 15
    n_theta = 30
    f_min = -3
    f_max = 3
    theta_min = -np.pi
    theta_max = np.pi
    #######################################################
    x = np.linspace(f_min, f_max, n_f)
    y = np.linspace(f_min, f_max, n_f)
    theta = np.linspace(theta_min, theta_max, n_theta)

    error_state[2] = np.mod(error_state[2] + np.pi, 2 * np.pi) - np.pi

    all_error_states = np.array(list(product(x, y, theta)))
    # check which index in the vectorized states is the closest to the error state
    diff = np.linalg.norm(all_error_states - (error_state + cur_ref), axis=1)
    count = np.argmin(diff)
    
    u = P_error_state[cur_iter, count]
    return u
    

    

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
    P_error_state = np.load('P_error_state_sd_nt100_nf15.npy')
    print(P_error_state.shape)
    P_error_state = P_error_state.reshape((100, -1, 2))

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
        control = new_gpi_controller(cur_state, cur_ref, cur_iter, P_error_state)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=False)
        print("cur_state", cur_state)
        print('cur_ref', cur_ref)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print("----------------------------------")
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
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
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

