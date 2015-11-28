"""
The Amygdala class.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tools

class Amygdala(object):
    """
    Assign reward to the appropriate features and track it over time.

    Functional MRI studies show that the amygdala participates in 
    the assignment or reward and punishment. It does lots of other things
    too and probably behaves nothing like this class.

    Attributes
    ----------
    cumulative_reward : float
        The total reward amassed since the last visualization.
    recent_features : list of arrays of floats
        This is the list of length ``trace_length`` containing the 
        most recently observed arrays of feature activities.
    recent_rewards : list of floats
        This is the list of length ``trace_length`` containing the 
        most recently observed reward values.
    reward_by_feature : array of floats
        The reward typically associated with each feature. The array has
        as many elements as there are features.
    reward_history : list of floats
        A time series of reward accumulated during the periods between
        each time the ``Amygdala`` is visualized..
    reward_learning_rate : float
        The upper bound of the rate at which the reward value for a feature
        is adjusted at each time step. Adjust this higher if you need the 
        brain to learn the feature values faster. Adjust it 
        lower if you want the estimates to be less noisy.
    reward_steps : list of ints
        A time series of the brain's age in time steps corresponding
        to each of the rewards in ``reward_history``.
    time_constant : float
        A value that scales the length of the trace. Adjust this if you
        believe the trace is too short or too long.
    trace_length : int
        The length of the window over which reward is integrated 
        in time steps. The weight of the rewards decay exponentially
        the older they are. 
    trace_magnitude : float
        The maximum magnitude of integrated reward. This is used to scale
        the reward trace such that the highest it can get is 1 and the
        lowest it can get is -1. 
    time_since_reward_log : int
        Number of time steps since reward was last logged. It gets
        logged every time the ``Amygdala`` is visualized.
    """

    def __init__(self, num_features):
        """
        Set up the ``Amygdala``.

        Parameters
        ----------
        num_features : int
            The number of features in the ``brain``. The ``Amygdala`` will 
            learn the reward associated with each of these.
        """    
        self.reward_by_feature = np.zeros(num_features)

        # Keep a recent history of reward and active features 
        # to account for delayed reward.
        # TODO: clean up Amygdala.
        self.time_constant = .5
        self.reward_learning_rate = 1e-2
        self.trace_length = int(12. * self.time_constant)
        self.trace_length = 1
        self.recent_rewards = list(np.zeros(self.trace_length))
        self.recent_features = [np.zeros(num_features)] * self.trace_length
        """
        The trace magnitude is the sum of largest possible magnitudes
        of all the reward values in the trace. Having it pre-calcuated
        helps to condense the trace into a single value. 
        Trace is calculated by working from the beginning of the list
        (the oldest) to the end (the most recent). Reward values are 
        decayed the further they are away in time from when the
        feature was active, up to ``trace_length`` time steps ago. 
        The decay function here is hyperbolic, rather than exponential. 
        It has a basis in both experimental psychology and economics 
        as representing typical human behavior.

        The math behind the function is just a fraction
        where time is on the bottom:
            decay = c / (c + time)
            where 
                decay is a fraction
                c is a time constant, typically a few time steps
                time is the number of time steps between the feature
                    being active and the reward being received
        """
        weights =  (1. / 
                    (1. + np.arange(self.trace_length) * self.time_constant) )
        self.trace_magnitude = np.sum(weights)
        # Track the reward gathered over the lifetime of the ``brain``.
        self.cumulative_reward = 0
        self.time_since_reward_log = 0 
        self.reward_history = []
        self.reward_steps = []

    def learn(self, new_features, reward):
        """
        Assign reward to features.
        
        Parameters
        ----------
        new_features : array of floats
            The most recently observed set of feature activities.
        reward : float
            The most recently observed reward value.
        """
        # Clip the reward so that it falls between -1 and 1.
        reward = np.maximum(np.minimum(reward, 1.), -1.)
        weights =  (1. / 
                    (1. + np.arange(self.trace_length) * self.time_constant) )
        weighted_trace = np.array(self.recent_rewards) * weights
        reward_trace = np.sum(weighted_trace)
        reward_trace /= self.trace_magnitude
        features = self.recent_features[0]

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
                                   features * self.reward_learning_rate)
                                   #features ** 2 * self.reward_learning_rate)

        # Update the activity, action history, and reward.
        self.recent_rewards.append(reward)
        self.recent_rewards.pop(0)
        self.recent_features.append(np.copy(new_features))
        self.recent_features.pop(0)

        # Log the reward.
        self.cumulative_reward += reward
        self.time_since_reward_log += 1

    def grow(self, increment):
        """
        Grow the ``Amygdala``.

        Parameters
        ----------
        increment : int
            The number of features to add.
        """
        self.reward_by_feature = tools.pad(self.reward_by_feature, -increment)
        new_recent_features = []
        for recent_features in self.recent_features:
            new_recent_features.append(tools.pad(recent_features, -increment))
        self.recent_features = new_recent_features

    def visualize(self, timestep, brain_name, log_dir):
        """
        Update the reward history, create plots, and save them to a file.

        There are two plots that get created. They show:
            1. The lifetime reward history of the ``brain``.
            2. The reward values associated with each feature.

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
        # Check whether any time has passed since the last update.
        if self.time_since_reward_log > 0:
            # Update the lifetime record of the reward.
            self.reward_history.append(float(self.cumulative_reward) / 
                                       (self.time_since_reward_log + 1))
            self.cumulative_reward = 0    
            self.time_since_reward_log = 0
            self.reward_steps.append(timestep)

        performance = np.mean(self.reward_history)

        # Plot the lifetime record of the reward.
        fig = plt.figure(11111)
        plt.plot(self.reward_steps, self.reward_history, color=tools.copper,
                 linewidth=2.5)
        plt.gca().set_axis_bgcolor(tools.copper_highlight)
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
            plt.plot([0., value], [i,i], color=tools.copper, linewidth=5.,
                     solid_capstyle='butt')
        plt.plot([0.,0.],[0., self.reward_by_feature.size - 1.], 
                 color=tools.copper_shadow, linewidth=1.)
        plt.gca().set_axis_bgcolor(tools.copper_highlight)
        max_magnitude = np.max(np.abs(self.reward_by_feature))
        plt.gca().set_xlim((-1.05 * max_magnitude, 1.05 * max_magnitude))
        plt.gca().set_ylim((-1., self.reward_by_feature.size))
        plt.xlabel('Reward')
        plt.ylabel('Feature index')
        plt.title('{0} Amygdala'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'reward_by_feature_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')
        
        return performance
