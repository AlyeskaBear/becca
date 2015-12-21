"""
The Ganglia class
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import tools

class Ganglia(object):
    """
    Plan goals and choose actions for the ``brain``.

    The ``Ganglia`` class is analogous to the basal ganglia in the human brain,
    a collection if fascinating structures and circuits deep in one of
    the oldest parts of the brain. No one understands exactly what they
    do, but when they are damaged by trauma or disease, people lose 
    the ability to choose and execute actions well. I'm going pretty far out
    on a neuroscientific limb by suggesting that basal ganglionic functions
    work like this class, but it is my best guess right now and I'm going
    to run with it.

    Attributes
    ----------
    decay_rate : float
        The fraction by which goal magnitudes are decreased at each 
        time step.
    goals : array of floats
        Goals cover the entire state, both actions and features. An action
        goal is an intention to execute that action in the current
        time step. A feature goal temporarily boosts the reward 
        associated with that feature. This encourages making decisions
        that activate that feature.
    goal_scale : float
        The maximum value that can be assigned to goals.
    num_actions, num_features : int
        See docstrings for ``brain.py``.    
    num_elements : int
        The combined number of features and actions.
    """

    def __init__(self, num_features, num_actions):
        """
        Configure the ``ganglia``.
        
        Parameters
        ----------
        num_features, num_actions : int
            See docstring for ``brain.py``.
        """
        self.num_actions = num_actions
        self.num_features = num_features
        self.num_elements = self.num_actions + self.num_features
        self.goals = np.zeros(self.num_features)
        self.decay_rate = .3
        self.goal_scale = 1.
        self.goal_threshold = .01
        
    def decide(self, features, decision_values):
        """
        Choose which, if any, actions to take on this time step.

        Only one decision can be made per time step. A decision can be
        either to take an action or to set a feature as a goal.

        Parameters
        ----------
        decision_values : array of floats
            Scores representing the expected reward with taking each 
            action or choosing each feature as a goal.
        features : array of floats
            The current activity of each of the features. 

        Returns
        -------
        actions : array of floats
            The collection of actions to be executed on the current time step.
        self.goals.copy() : array of floats
            The goal values associated with each of the features.
        """
        # Decay goals with time.
        self.goals *= 1. - self.decay_rate

        # Let goals be fulfilled by their corresponding features.
        self.goals -= features
        self.goals = np.maximum(0., self.goals) 
        self.goals[np.where(self.goals < self.goal_threshold)] = 0.

        # TODO: Another option is to set the most recently attended
        # feature as the goal. This provides some value, which could
        # be set as a constant.

        # DEBUG
        #for i,dv in enumerate(decision_values):
        #    print ' '.join(['    ', str(i), ':', '{0:.3}'.format(dv) ])

        decisions = np.zeros(decision_values.shape)
        if True:
            decision_index = np.argmax(decision_values)
        else:
            rectified_values = np.maximum(decision_values, 0.) ** 8
            probabilities = rectified_values / np.sum(rectified_values)
            decision_index = np.random.choice(np.arange(decisions.size), 
                                              p=probabilities)
        #print 'decision values'
        #tools.format(decision_values)
        decisions[decision_index] = 1.
        #print 'di', decision_index
        actions = decisions[:self.num_actions]
        # TODO: Pass through predictable (well-learned) actions and goals. 

        # Add the decisions to the ongoing set of goals.
        self.goals = np.maximum(self.goals, 
                                decisions[self.num_actions:] *
                                self.goal_scale)
        # Choose a single random action. Used for debugging and testing.
        random_action = False
        if random_action:
            decisions = np.zeros(decisions.shape)
            decision_index = np.random.randint(self.num_elements)
            decisions[decision_index] = 1. 
            # In the state representation, actions come first.
            actions = decisions[:self.num_actions].copy()

        return actions, self.goals.copy()

    def grow(self, increment):
        """
        Grow the ``Ganglia``.

        Parameters
        ----------
        increment : int
            The number of features to add.
        """
        self.num_features += increment
        self.num_elements += increment
        self.goals = tools.pad(self.goals, self.num_features)

    def visualize(self, brain_name, log_dir):
        """
        Make pictures that describe the state of the ``Ganglia``.

        Parameters
        ----------
        brain_name : str
            See docstring for ``brain.py``.
        log_dir : str
            See docstring for ``brain.py``.
        """
        #print 'ganglia:'
        #print '    goals:'
        #for i in np.arange(self.goals.size):
        #    print ' '.join(['        ', str(i), 
        #                    ': {0:.3}'.format(self.goals[i])]) 
            
        # Plot the learned reward value of each feature.
        fig = plt.figure(21122)
        fig.clf()
        for i, value in enumerate(self.goals):
            plt.plot([0., value], [i,i], color=tools.copper, linewidth=5.,
                     solid_capstyle='butt')
        plt.plot([0.,0.],[0., self.goals.size - 1.], 
                 color=tools.copper_shadow, linewidth=1.)
        plt.gca().set_axis_bgcolor(tools.copper_highlight)
        max_magnitude = np.max(np.abs(self.goals))
        plt.gca().set_xlim((-1.05 * max_magnitude, 1.05 * max_magnitude))
        plt.gca().set_ylim((-1., self.goals.size))
        plt.xlabel('Goals')
        plt.ylabel('Feature index')
        plt.title('{0} Ganglia'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'reward_by_feature_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')
        
