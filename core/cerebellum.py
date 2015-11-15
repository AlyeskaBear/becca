"""
The Cerebellum class
"""

import numba_tools as nb
import numpy as np
import matplotlib.pyplot as plt
import os
import tools
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Cerebellum(object):
    """
    Make a model of the world for choosing and evaluating actions and plans.

    Like BECCA's other functional models of brain regions, the exact role
    of the cerebellum is hotly contested. It's structure is more regular 
    and has been mapped in more detail than that of any other structure
    except perhaps the hippocampus. Here's an educated guess about
    what it's doing, but consider yourself warned: It doesn't exactly match
    anyone else's guess, including those of some extremely smart people
    who have been studying it for decades. The only consolation I have
    is that the smart people disagree with each other too. 

    Everyone agrees that the cerebellum has a big role in the control 
    and execution of coordinated  movements. I hypothesize that it does
    this by creating a model of the world. This is somewhat similar to
    other proposed cerebellar models, but this one is uniquely mine. 
    I can't blame its shortcomings on anyone else.

    At it's core, BECCA's cerebellum is a state-goal-state model 
    of the world that it builds based on its experience.
    The model is a collection of transition probabilities.
    A transition is of the form
        feature_1 -> goal -> feature_2
    that is, if feature_1 is followed by a given goal, then feature_2 
    is likely to occur. 
    A goal can be either an action or a feature.
    If a goal is selected that is an action, that action is executed.
    If a goal is selected that is a feature, that feature is temporarily
    assigned a reward. 
    Each transition is assigned a probability that represents its 
    likelihood.

    Attributes
    ----------
    curiosities,
    opportunities : 2D array of floats
    observations, 
    probabilities : 3D array of floats
        The properties associated with each transition, based 
        on BECCA's experience. If M is the number of features and 
        N is the number of actions,
        then the shape of 2D arrays is (M, N+M) and the shape of 
        3D arrays is (M, N+M, M). As a heads up, this can eat up 
        memory as M and N get large. They are indexed as follows:
            index 0 : feature_1 (past)
            index 1 : goal
            index 2 : feature_2 (future)
        The 2D arrays can be 2D because they inherently lack any
        information about what is going to happen next.:

    curiosity_rate : float
        The rate at which curiosities are accumulated each time step.
    goals : array of floats
        A history of the index of the state element chose as a goal. 
    time_constant : float
        The number of time steps into the pats or future in which 
        the perceived magnitude decreases to half of its original value.
    trace_length : int
        The length of the history used, both into the past and into the
        future. 
    """
    
    def __init__(self, num_features, num_actions):
        """
        Configure the ``cerebellum``.
        
        Parameters
        ----------
        num_features, num_actions : int
            See docstring for ``brain.py.num_actions``.
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.num_elements = num_actions + num_features 
        _2D_size = (self.num_features, self.num_elements)
        _3D_size = (self.num_features, self.num_elements, self.num_features)
        self.observations = np.zeros(_3D_size)
        self.opportunities = np.ones(_2D_size) * tools.EPSILON
        self.probabilities = np.zeros(_3D_size)
        self.curiosities = np.zeros(_2D_size)
        self.curiosity_rate = 1e-1 / float(self.num_elements)
        self.time_constant = 1.
        self.trace_length = int(100. * self.time_constant)
        self.goal_magnitude = .3

        # The 0th array of features is the context.
        # The 1st through end (tracel_length+1) arrays are the result trace.
        self.features_history = ([np.zeros(self.num_features)] 
                                 * (self.trace_length + 1))
        self.goals_history = [None] * (self.trace_length + 1)

    def get_decision_values(self, features, feature_reward, goal_reward):
        """
        Rate each of the potential decisions, based on their expected reward.
        
        This is takes a lot of cycles to compute, so it was streamlined
        using numba. The numba function does most of the heavy lifting 
        here. Refer to it for more complete documentation on the method.

        Parameters
        ----------
        features : array of floats
            The current activities of all the features.
        feature_reward : array of floats
            The expected reward associated with each of the features.
            Note that this doesn't include actions. They aren't 
            tracked in the ``Amygdala``. It can also be negative.
        goal_reward : array of floats
            The temporary reward associated with each of the features
            due to each being recently chosen as a decision. Note that 
            this also doesn't include actions, and can also be negative.

        Returns
        -------
        decision_values : array of floats
            These are an array of expected values, one for each of the 
            potential decisions that the ``Ganglia`` can choose. 
        """
        # Add together the observed reward associated with each feature
        # with the goal reward associated with each feature.
        total_reward = feature_reward + goal_reward
        # We're only interested in pursuing positive rewards.
        total_reward[np.where(total_reward < 0.)] = 0.
                         
        decision_values = np.zeros(self.num_elements)
        # Call a numba routine to make this calculation fast.
        nb.get_decision_values(self.probabilities, 
                               self.curiosities, 
                               features, 
                               total_reward, 
                               decision_values)
        # Decision_values should all be between 0 and 1.
        return decision_values


    def learn(self, features, actions, feature_goals):
        """
        Update the transition model of the world.

        Parameters
        ----------
        features : array of floats
            The current activities of the features.
        actions : array of floats
            The most recent set of commanded actions.
        feature_goals : array of floats
            The current set of goal values associated with each feature.
        """
        current_context = features.copy()
        goals = np.concatenate((actions, feature_goals))

        # Add the newest features and goal to the trace history.
        self.features_history.append(current_context)
        self.goals_history.append(goals)
        # Drop the oldest feature and goal values. They are now outdated. 
        # This restores the histories to their proper lengths.
        self.features_history.pop(0)
        self.goals_history.pop(0)
        # Update transition properties based on the goal 
        # and attended element history.
        training_goals = self.goals_history[0]
        if training_goals is not None:
            training_context = self.features_history[0] 

            # The results are the maximum value of each feature over
            # the time-decayed trace.
            decay = (1. / np.cumsum(np.ones(
                    self.time_constant * self.trace_length)))
            trace = np.array(self.features_history[1:])
            decayed_trace = trace * decay[:,np.newaxis]
            training_results = np.max(decayed_trace, axis=0)

            # Call a numba routine to make this calculation fast.
            nb.cerebellum_learn(self.opportunities, self.observations,
                                self.probabilities, self.curiosities,
                                training_context, training_goals,
                                training_results, current_context, goals,
                                self.curiosity_rate)


    '''
    # TODO: For multi-step planning simulate the likely effect of actions
    # into the future.
    def simulate(self, actions):
        """
        Imagine what would happen if ``actions`` were taken.

        Parameters
        ----------
        actions : array of floats
            The most recent set of actions selected.

        Returns
        -------
        """
        pass
    '''

    def grow(self, increment):
        """
        Grow the ``Cerebellum``.

        Parameters
        ----------
        increment : int
            The number of features to add.
        """
        self.num_elements += increment
        self.num_features += increment
        _2D_size = (self.num_features, self.num_elements)
        _3D_size = (self.num_features, self.num_elements, self.num_features)
        self.observations = tools.pad(self.observations, _3D_size)
        self.opportunities = tools.pad(self.opportunities, _2D_size, 
                                       val=tools.EPSILON)
        self.probabilities = tools.pad(self.probabilities, _3D_size)
        self.curiosities = tools.pad(self.curiosities, _2D_size)

        for i in np.arange(len(self.features_history)):
            self.features_history[i] = tools.pad(self.features_history[i], 
                                                 self.num_features)
        for i in np.arange(len(self.goals_history)):
            if self.goals_history[i] is not None:
                self.goals_history[i] = tools.pad(self.goals_history[i], 
                                                  self.num_elements)

    def visualize(self, brain_name, log_dir):
        """
        Represent the state of the ``cerebellum`` in pictures.
        
        Parameters
        ----------
        brain_name : str
            See docstring for ``brain.py``.
        log_dir : str
            See docstring for ``brain.py``.
        """
        max_prob = np.max(self.probabilities)
        fig = plt.figure(num=777777)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        indices = np.where(self.probabilities > .4 * max_prob)
        ax.scatter(indices[0], indices[1], zs=indices[2], 
                   zdir=u'z', s=5, c=tools.DARK_COPPER, depthshade=True) 
        ax.set_xlabel('Past elements')
        ax.set_ylabel('Goals')
        ax.set_zlabel('Future elements')
        # Adjust azim to look at the plot from different directions.
        ax.azim = 250
        plt.title('Cerebellum transitions'.format(brain_name))
        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'cerebellum_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')

        fig = plt.figure(num=777778)
        plt.clf()
        ax = plt.subplot(2,3,1)
        plt.gray()
        plt.imshow(self.opportunities, interpolation="nearest")
        ax.set_xlabel(' '.join(['max', str(np.max(self.opportunities))]))
        plt.title('Cerebellum opportunities {0}'.format(brain_name))

        ax = plt.subplot(2,3,2)
        plt.gray()
        plt.imshow(np.sum(self.observations,axis=2), interpolation="nearest")
        ax.set_xlabel(' '.join(['max', str(np.max(self.observations))]))
        plt.title('observations')

        #ax = plt.subplot(2,3,3)
        #ax = plt.subplot(2,3,4)

        ax = plt.subplot(2,3,5)
        plt.gray()

        ax = plt.subplot(2,3,6)
        plt.gray()
        plt.imshow(self.curiosities, interpolation="nearest")
        ax.set_xlabel(' '.join(['max', str(np.max(self.curiosities))]))
        plt.title('curiosities')

        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'cerebellum_{0}.png'.format(brain_name)
        pathname = os.path.join(log_dir, filename)
        plt.savefig(pathname, format='png')
