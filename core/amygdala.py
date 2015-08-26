"""
The Amygdala class.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class Amygdala(object):
    """
    Assign reward to the appropriate features and track it over time.

    Attributes
    ----------
    reward : float
        See docstring for ``brain.py.reward``.
    reward_by_feature : array of floats
        The reward typically associated with each feature. The array has
        as many elements as there are features.
    time_since_reward_log : int
        Number of time steps since reward was last logged. It gets
        logged every time the amygdala is visualized.
    cumulative_reward : float
        The total reward amassed since the last visualization.
    reward_history : list of floats
        A time series of reward accumulated at each visualization.
    reward_steps : list of ints
        A time series of the brain's age in time steps corresponding
        to each of the rewards in ``reward_history``.
    """

    def __init__(self, num_features):
        """
        Set up the Amygdala.
        """    
        self.reward_by_feature = np.zeros(num_features)

        # Keep a recent history of reward and active features 
        # to account for delayed reward.
        self.TIME_FACTOR = 1.
        self.REWARD_LEARNING_RATE = 1e-3
        self.TRACE_LENGTH = int(self.TIME_FACTOR * 6.)
        self.trace_magnitude = 0.
        for tau in np.arange(self.TRACE_LENGTH):
            self.trace_magnitude += 2. ** (-self.TIME_FACTOR * float(tau))
        self.recent_rewards = list(np.zeros(self.TRACE_LENGTH))
        self.feature_history = [np.zeros(num_features)] * self.TRACE_LENGTH
        #self.action_history = [np.zeros(num_features)] * (
        #        self.TRACE_LENGTH + 1)

        # Track the reward gathered over the lifetime of the ``brain``.
        self.cumulative_reward = 0
        self.time_since_reward_log = 0 
        self.reward_history = []
        self.reward_steps = []

    def step(self, new_features, reward):
        """
        Assign reward to features.
        """
        # Clip the reward so that it falls between -1 and 1.
        reward = np.maximum(np.minimum(reward, 1.), -1.)

        # Update the reward trace, a decayed sum of recent rewards.
        reward_trace = 0.
        for tau in range(self.TRACE_LENGTH):
            """
            Work from the end of the list, from the most recent
            to the oldest, decaying future values the further
            they are away from the cause and effect that occurred
            TRACE_LENGTH time steps ago. The decay function here
            is hyperbolic, rather than exponential. It has a basis
            in both experimental psychology and economics as representing
            typical human behavior.
            """
            reward_trace += self.recent_rewards[tau] * (
                    2. ** (-self.TIME_FACTOR * float(tau)))
        reward_trace /= self.trace_magnitude
        features = self.feature_history[0]

        """
        Increment the expected reward value associated with each feature.
        The size of the increment is larger when:
            1. the discrepancy between the previously learned and 
                observed values is larger and
            2. the feature activity is greater.
        Another way to say that is if either the discrepancy is very small
        or the feature activity is very small, there is no change.
        """
        self.reward_by_feature += ((reward_trace - self.reward_by_feature) * 
                                   features * self.REWARD_LEARNING_RATE)

        # Update the activity, action history, and reward.
        self.recent_rewards.append(reward)
        self.recent_rewards.pop(0)
        self.feature_history.append(np.copy(new_features))
        self.feature_history.pop(0)
        #self.action_history.append(np.copy(goal))
        #self.action_history.pop(0)

        # Log the reward.
        self.cumulative_reward += reward
        self.time_since_reward_log += 1

    def visualize(self, timestep, brain_name, log_dir):
        """
        Update the reward history, show it in a plot, and save it to a file.

        Parameters
        ----------
        timestep : int
            See docstring for ``brain.py``.
        brain_name : str
            See docstring for ``brain.py``.
        log_dir : str
            See docstring for ``brain.py``.

        Returns
        -------
        performance : float
            The average reward over the lifespan of the ``brain``.
        """
        # Update the lifetime record of the reward.
        self.reward_history.append(float(self.cumulative_reward) / 
                                   (self.time_since_reward_log + 1))
        performance = np.mean(self.reward_history)
        self.cumulative_reward = 0    
        self.time_since_reward_log = 0
        self.reward_steps.append(timestep)

        # Plot the lifetime record of the reward.
        fig = plt.figure(11111)
        plt.plot(self.reward_steps, self.reward_history)
        plt.xlabel('Time step')
        plt.ylabel('Average reward')
        plt.title('Reward history for {0}'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'reward_history_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')

        # Plot the learned reward value of each feature.
        fig = plt.figure(11112)
        fig.clf()
        for i, value in enumerate(self.reward_by_feature):
            print i, value
            plt.plot([i,i], [0., value], color='green', linewidth=5.)
        plt.gca().set_ylim((-1., 1.))
        plt.gca().set_xlim((-1., self.reward_by_feature.size + 1.))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'reward_by_feature_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')
        
        return performance
