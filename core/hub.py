""" the Hub class """
import matplotlib.pyplot as plt
import numpy as np
import tools

class Hub(object):
    """ 
    The central long term memory and action selection mechanism 
    
    The analogy of the hub and spoke stucture is suggested by 
    the fact that the hub has a connection to each
    of the gearboxes. In the course of each timestep it 
    1) reads in a copy of the input cable activities to each of the gearboxes
    2) updates the reward estimate for current transitions
    3) selects a goal and
    4) declares that goal in the appropriate gearbox.
    """
    def __init__(self, initial_size, num_actions=0, num_sensors=0, name='hub'):
        self.name = name
        self.num_cables = initial_size
        # Set constants that adjust the behavior of the hub
        self.REWARD_LEARNING_RATE = 1e-2
        # Keep a history of reward and active features to account for 
        # delayed reward.
        self.TRACE_LENGTH = 25
        self.FORGETTING_RATE = 1e-6
        # As time passes, grow more optimistic about the effect of 
        # trying goals.
        self.ROSY_GLASSES = .001#0.#.1
        self.COUNT_FACTOR = 3.#.3
        # Calculate initial reward in order to encourage exploration
        self.INITIAL_REWARD = .2
        self.TIME_FACTOR = 10.  
        self.trace_magnitude = 0.
        for tau in np.arange(self.TRACE_LENGTH):
            self.trace_magnitude += 1. / (1. + self.TIME_FACTOR * float(tau))

        # Initialize variables for later use
        self.reward_history = list(np.zeros(self.TRACE_LENGTH))
        feature_shape = (self.num_cables, 1)
        self.activity_history = [np.zeros(feature_shape)] * (
                self.TRACE_LENGTH + 1)
        self.action_history = [np.zeros(feature_shape)] * (
                self.TRACE_LENGTH + 1)
        self.mask = np.zeros(feature_shape)
        self.mask[:num_actions + num_sensors] = 1.
        # reward is a property of every transition. 
        # In this 2D array representation, each row
        # represents a feature, and each column represents 
        # a goal (action). Element [i,j] represents the transition
        # from feature i to action j.
        transition_shape = (self.num_cables, self.num_cables)
        self.reward = np.ones(transition_shape) * self.INITIAL_REWARD
        # This counts the number of times each transition has been encountered 
        self.count = np.zeros(transition_shape)
        self.frame_counter = 10000

    def step(self, cable_activities, raw_reward):
        """ Advance the hub one step """
        # Update the reward trace, a decayed sum of recent rewards.
        reward_trace = 0.
        for tau in range(self.TRACE_LENGTH):
            # Work from the end of the list, from the most recent
            # to the oldest, decaying future values the further
            # they are away from the cause and effect that occurred
            # TRACE_LENGTH time steps ago.
            reward_trace += self.reward_history[tau] / (
                    1. + self.TIME_FACTOR * float(tau))
        reward_trace /= self.trace_magnitude

        self.mask = np.sign(np.maximum(self.mask, cable_activities))
        state = self.activity_history[0]
        action = self.action_history[0]

        # Increment the count according to the most recent features and actions
        self.count += state * action.T 
        # Decay the count gradually to encourage occasional re-exploration 
        self.count *= 1. - self.FORGETTING_RATE
        #self.count = np.maximum(self.count, 0)
        # Update the expected reward
        # To avoid unnecessary computation, check whether there has been an 
        # action take before bothering to update the reward.
        if np.where(action != 0.)[0].size:
            age_factor = 1. / (1. + self.TIME_FACTOR * self.count)
            # Calculate the rate at which to update the reward estimate
            rate = ((1. - self.REWARD_LEARNING_RATE) * age_factor + 
                    self.REWARD_LEARNING_RATE)
            optimism = ((self.mask * (1. - self.reward) * self.mask.T *
                         self.ROSY_GLASSES * age_factor) / 
                        (self.num_cables ** 2))
            # Use the square of the state (feature activities) in order
            # de-emphasize partially excited states. Focus the adaptation
            # on cases where the state is most activated.
            self.reward += ((reward_trace - self.reward) * 
                            (state * state * action.T) * 
                            rate) + optimism
        # Choose a goal at every timestep
        #goal = np.zeros((self.num_cables, 1))
        state_weight = cable_activities 
        weighted_reward = state_weight * self.reward
        #expected_reward = (np.sum(weighted_reward, axis=0) / 
        #                   np.sum(state_weight))
        expected_reward = (np.max(weighted_reward, axis=0) + 
                           np.min(weighted_reward, axis=0))
        expected_reward[np.where(self.mask.ravel() < 1.)] -= 2.
        best_reward = np.max(expected_reward)
        potential_winners = np.where(expected_reward == best_reward)[0] 
        # Break any ties by lottery
        if potential_winners.size > 0:
            goal_cable = potential_winners[np.random.randint(
                    potential_winners.size)]
        else:
            goal_cable = np.random.randint(self.num_cables)

        # Only recommend a goal if it is expected to bring a greater 
        # reward than what is currently being experienced.
        #return goal_cable, best_reward
        if best_reward > raw_reward * self.trace_magnitude:
            return goal_cable, best_reward
        else:
            return None, 0.
       
    def update(self, cable_activities, issued_goal_index, raw_reward): 
        """ Assign the goal to train on, based on the goal that was issued """
        goal = np.zeros((self.num_cables, 1))
        if issued_goal_index is not None:
            goal[issued_goal_index] = 1.
        # Update the activity, action history, and reward.
        self.reward_history.append(raw_reward)
        self.reward_history.pop(0)
        self.activity_history.append(np.copy(cable_activities))
        self.activity_history.pop(0)
        self.action_history.append(np.copy(goal))
        self.action_history.pop(0)
        return

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new gearboxes are created """ 
        self.num_cables = self.num_cables + num_new_cables
        features_shape = (self.num_cables, 1)
        transition_shape = (self.num_cables, self.num_cables) 
        self.reward = tools.pad(self.reward, transition_shape,
                                val=self.INITIAL_REWARD)
        #self.reward = tools.pad(self.reward, transition_shape, val=0.)
        self.count = tools.pad(self.count, transition_shape)

        self.mask = tools.pad(self.mask, features_shape)
        for index in range(len(self.activity_history)):
            self.activity_history[index] = tools.pad(
                    self.activity_history[index], features_shape)
            self.action_history[index] = tools.pad(
                    self.action_history[index], features_shape)
