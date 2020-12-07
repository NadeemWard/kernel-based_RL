import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import normalize
import sys
import cvxpy as cp
import pdb

def get_data(env, total_samples_per_action=1000, random = True, V=None, R=None, kernel = None,
             data = None, gamma = None):
    '''
    Function to collect data at random from OpenAI gym type environment. Use for CartPole-v0 primarily
    :param env: gym type env
    :param total_samples_per_action: number of samples to collect per action
    :return: - return the observed transitions per action, all concatentated into a large matrix ("transition_data")
             - return the rewards observed from those transitions ("reward_data")

             transition_data is of the form ( num_samples X num_actions X 2 (for starting state and next state) X state dimension )
             reward_data is of the form: ( num_samples X num_actions)

    '''

    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    transition_data = np.zeros([total_samples_per_action, num_actions, 2, state_dim])  # placeholder for data to store
    reward_data = np.zeros([total_samples_per_action, num_actions])

    num_samples_per_action = np.zeros(num_actions)
    while min(num_samples_per_action) < total_samples_per_action:
        # run another full episode
        x = env.reset()
        done = False
        while not done:

            if random:
                action = env.action_space.sample()
            else:
                action = get_action(V, R, kernel, data, gamma, x)

            next_x, reward, done, _ = env.step(action)

            if num_samples_per_action[action] < total_samples_per_action:
                transition_data[int(num_samples_per_action[action]), action, 0, :] = x
                transition_data[int(num_samples_per_action[action]), action, 1, :] = next_x

                reward_data[int(num_samples_per_action[action]), action] = reward

                num_samples_per_action[action] += 1

            # update current state
            x = next_x

            if done:
                reward_data[int(num_samples_per_action[action]) - 1, action] = 0 # change the reward to be zero

    return transition_data, reward_data


def kernel_matrix(X_s, Y_s, kernel):
    '''
    X_s: data matrix of initial states of size (num_samples, num_dimensions_state_space)
    Y_s: data matrix of next states of size (num_samples, num_dimensions_state_space)

    These data matrices are both for a specific action a

    K: return the kernel matrix of the cross product between elements of these two matrices, size (num_samples, num_samples)
    using the gaussian kernel. Element i,j of this matrix is kernel([X_s]_i, [Y_s]_j)
    '''

    m, dim_s = X_s.shape

    return normalize(kernel(X_s, Y_s), axis=0,
                     norm="l1")  # normalize the kernel values along axis 0 to have them sum to 1


def kernel_tensor(X, Y, kernel):
    '''
    X: data tensor of initial states of size (num_samples, num_actions, num_dim_state_space)
    Y: data tensor of next states of size (num_samples, num_actions, num_dim_state_space)

    return: K, the kernel tensor of concatenated kernel matrices of each action seperately
    '''

    num_samples, num_actions, dim_s = X.shape
    K = np.zeros((num_samples, num_actions, num_samples))

    for a in range(num_actions):
        K[:, a, :] = kernel_matrix(X[:, a, :], Y[:, a, :], kernel)  # get the kernel matrix per action

    return K


def get_action(V, R, kernel, data, gamma, x):
    '''
    V: the value function, a matrix of size (num_samples, num_actions)
       for each "next state" seen in the data
    R: matrix of 1 step rewards of size (num_samples, num_actions)
    kernel: the kernel function used
    data: the data tensor or size (num_samples X num_actions X 2 (x_s, y_s) X dim(state_space) )
    gamma: discount factor
    bandwidith: hyperparameter for kernel function
    x: the actual state we are evaluating

    return: indx of action to take
    '''

    num_samples, num_actions = V.shape

    Q = np.zeros(num_actions)
    for i in range(num_actions):
        X_a = data[:, i, 0, :]  # shape (num_samples, dim_state_space)
        Q[i] = np.dot(normalize(kernel(X_a, x.reshape(1, -1)), axis=0, norm="l1").T, R[:, i] + gamma * V[:, i])

    return np.argmax(Q)


def value_iteration(Theta, R, gamma, stopping_criteria=10e-5, axis=2):
    '''
    Theta: Tensor of kernel values for the data of size (num_samples, num_actions, num_samples)
    R: one step rewards observed of size (num_samples, num_actions)
    gamma: discount factor
    num_iterations: number of times we want to iterate the algorithm

    return: The new Value functions we get; of size (num_samples, num_actions)
    '''

    num_samples, num_actions = R.shape
    V_old = np.zeros((num_samples, num_actions))

    abs_error = sys.maxsize
    num_iterations = 0
    while abs_error > stopping_criteria:

        # compute the Q value
        Q = np.zeros((num_samples, num_actions, num_actions))
        for i in range(num_actions):
            Q[:, i, :] = np.dot(Theta[:, i, :].T, R + gamma * V_old)

        # do max over axis
        V = np.amax(Q, axis=axis)

        # compute error (largest absolute difference)
        abs_error = np.max(np.abs(V_old - V))

        V_old = V
        num_iterations += 1

    # print("Number of iterations of value iterations until convergence:", num_iterations)
    return V


def different_value_iteration(X, Y, R, kernel, gamma, stopping_criteria=10e-5):
    '''
    My implementation with kernel computation between action datasets S^a. This is just to make sure Im doing the computation
    right.
    :param X: starting state data of the form num_samples X num_actions X
    :param Y: next state data of the form num_samples X num_actions X
    :param R:  One step rewards of the form num_samples X num_actions
    :param gamma: discount factor
    :param stopping_criteria: When we will terminate value iteration
    :return: Return the value functions found
    '''

    num_samples, num_actions = R.shape
    V_old = np.zeros((num_samples, num_actions))

    abs_error = sys.maxsize
    num_iterations = 0

    while abs_error > stopping_criteria:

        td_update = R + gamma * V_old
        V = np.zeros((num_samples, num_actions))
        for sample_indx in range(num_samples):
            for action_indx in range(num_actions):

                # loop over each action-value function
                Q_x = np.zeros(num_actions)
                for a in range(num_actions):
                    Q_x[a] = np.dot(normalize(kernel(X[:, a, :], Y[sample_indx, action_indx, :].reshape(1, -1)), axis=0,
                                              norm="l1").T, td_update[:, a])

                V[sample_indx, action_indx] = max(Q_x)

        # compute error (largest absolute difference)
        abs_error = np.max(np.abs(V_old - V))

        V_old = V
        num_iterations += 1

    print("Number of iterations of value iterations until convergence:", num_iterations)
    return V


def test_kbrl_env(env, V=None, R=None, kernel=None, gamma=None, data=None, num_episodes=1, random=False):
    '''
    Getting test performance. What this code does is loop num_episodes times over the env and saves all the
    rewards received per episode
    :param env: env used
    :param V: the value function found from {value iteration / linear programming } needed for action selection
    :param R: one step rewards (needed for action selection)
    :param kernel: kernel function used (needed for aciton selection)
    :param gamma: discount factor
    :param data: data generated
    :param num_episodes: number of iterations

    :return: returns all of the episode rewards received
    '''

    rewards = []
    for i in range(num_episodes):

        episode_reward = 0
        num_steps = 0

        done = False
        state = env.reset()
        while not done:

            if random:
                action = env.action_space.sample()
            else:
                action = get_action(V, R, kernel, data, gamma, state)

            state, reward, done, _ = env.step(action)

            episode_reward += reward
            num_steps += 1

        rewards.append(episode_reward)

    return np.array(rewards)


def plot_results(env, transition_data, reward_data, kernel_vals, gamma_vals,
                 num_episodes=10, axis = 2, lp=False, path = None):
    '''
    Plotting function. Putting everything together

    :param env: env used
    :param transition_data: transition dynamics
    :param reward_data: reward data
    :param kernel_vals: the different hyperparmeters for the RBF kernel to try
    :param gamma_vals: different gamma values to try
    :param num_episodes: number of episode we want to average performance over
    :param axis: how to maximize in policy iteration
    :param lp: wheter to solve using LP approach or not
    :param path: path to save model to. If None won't save.
    :return: None. Just plot the result
    '''

    num_samples_per_action, num_actions = reward_data.shape

    X = transition_data[:, :, 0, :]  # num_samples_per_action, num_actions, dim_state
    Y = transition_data[:, :, 1, :]  # num_samples_per_action, num_actions, dim_state

    for gamma in gamma_vals:

        rewards = []
        for b in kernel_vals:
            # define kernel
            kernel = RBF(b)

            # compute kernel tensor
            Theta = kernel_tensor(X, Y, kernel)

            # compute value iteration
            if lp:
                init_dist = np.ones((num_samples_per_action, num_actions)) * 1 / num_samples_per_action
                V = kblp(Theta=Theta, R=reward_data, gamma=gamma, initial_dsitribution=init_dist)
            else:
                V = value_iteration(Theta, reward_data, gamma=gamma, stopping_criteria=10e-3, axis= axis)
            # V = different_value_iteration(X, Y, reward_data, kernel = kernel, gamma = gamma)

            # save model
            if path:
                np.savez(path + "/data_gamma=" + str(gamma)+"_b=" + str(b), V = V,
                        transition_data = transition_data, reward_data = reward_data)

            # run on test environement
            rewards.append(test_kbrl_env(env, V=V, R=reward_data, kernel=kernel, gamma=gamma,
                                         data=transition_data, num_episodes=num_episodes, random=False))

        # rewards will a matrix of size num_kernel_vals X num_cummulative_rewards_per_episode
        rewards = np.array(rewards)
        average = rewards.mean(axis=1)
        sigma = rewards.std(axis=1)

        # save results
        if path:
            np.savez(path + "/results", rewards = rewards, average = average, sigma = sigma)

        plt.plot(kernel_vals, average, label="gamma = {0}".format(gamma))
        plt.fill_between(kernel_vals, average + sigma, average - sigma, alpha=0.5)

        # plt.plot(kernel_vals, [random_reward] * len(kernel_vals), label="Random Agent")

    plt.xlabel("bandwidth value")
    plt.ylabel("Average reward")
    # plt.title("Average performance for different values of the bandwidth parameter")
    plt.legend()

    plt.show()


def kblp(Theta, R, gamma, initial_dsitribution):
    '''
    LP implementation using kernel based RL

    :param Theta: kernel Tensor
    :param R: reward data
    :param gamma: discount factor
    :param initial_dsitribution: the weighting in the objective function
    :return: the optimal value function
    '''
    # information about samples
    num_samples, num_actions = R.shape

    # define variables
    v = cp.Variable((num_samples, num_actions))

    # create objective function
    objective = cp.Minimize(cp.trace(initial_dsitribution.T @ v))

    # create constraints
    constraints = [v >= Theta[:, a, :].T @ (R + gamma * v) for a in range(num_actions)] # axis = 1

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose = False)

    return v.value


if __name__ == "__main__":

    # Define env
    env = gym.make("CartPole-v0")
    gamma = 0.99
    num_actions = env.action_space.n

    # get data
    num_samples_per_action = 1500
    transition_data, reward_data = get_data(env, total_samples_per_action=num_samples_per_action)


    X = transition_data[:, :, 0, :]  # num_samples_per_action, num_actions, dim_state
    Y = transition_data[:, :, 1, :]  # num_samples_per_action, num_actions, dim_state

    #####################################################################################################
    #################################### Value Iteration Approach #######################################
    #####################################################################################################

    # define kernel values to try
    kernel_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2]
    gamma_vals = [0.99]
    #plot_results(env, transition_data, reward_data, kernel_vals, gamma_vals, num_episodes=1000, save_name="KBRL_test2")

    #####################################################################################################
    ############################################## LP approach ##########################################
    #####################################################################################################
    plot_results(env, transition_data, reward_data, kernel_vals, gamma_vals, num_episodes = 1000, lp = True, save_name="LP_test9") # takes a while