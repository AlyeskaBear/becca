""" 
the Arborkey class 
"""
import numpy as np

class Arborkey(object):
    """ 
    Compares potential goals and decides which to send to the drivetrain
    
    The arborkey is at the highest level of control in the agent.
    It is named for the toothed key used to clamp drill chuck jaws
    around a bit. The arbor key determines which tool is used
    and in which orientation. It is also an intentional reference to the 
    key used to wind a clockwork mechanism,
    due to the fact that it is indispensible and ultimately
    controls whether the mechanism does anything useful.
    
    In the course of each timestep the arborkey
    decides whether to 
    1) pass on the hub's newest goal
    2) pass on a previous hub goal or
    3) don't pass any new goals 
    to the drivetrain.
    
    As it matures, it will also modulate its level of arousal--
    how long it allows itself to evaluate options before taking action--
    based on its recent reward and punishment history.
    
    TODO: Incorporate restlessness. The longer the arborkey goes
    without taking an action, the more likely it will be to act.
    """
    def __init__(self, num_cables):
        # The number of elements stored in short term memory
        # TODO: change name
        # debug: remove list of goals? 
        # Maybe the most recent is always the one chosen.
        #self.STM_LENGTH = 25
        self.STM_LENGTH = 1

        # Propensity to act
        self.restlessness = 0.
        # Amount by which restlessness is incremented each time step
        # debug: increased from 1e-5 to 1e-4
        self.NERVOUSNESS = 1e-4
        self.goal_candidates = []
        self.expected_reward = []
        self.time_since_observed = []

    def step(self, goal_candidate, hub_reward, curiosity,
             candidate_reward, current_reward):
        """ 
        Advance the arborkey through one time step.
        Evaluate goal candidates and possibly recommend one. 
        """
        """
        Increment restlessness by a small amount for each time step in which 
        an action is not taken. This helps the agent avoid getting stuck
        in ruts. 
        """
        self.restlessness += self.NERVOUSNESS * (1. - current_reward)
        goal = None
        # Update the list of goal candidates 
        #print 'gc', goal_candidate,'hr', hub_reward, 'c', curiosity, 'r', self.restlessness, 'cr', current_reward, 'sum', hub_reward + curiosity + self.restlessness
        if goal_candidate is not None:
            #if hub_reward + curiosity + self.restlessness >= current_reward:
            if True:
                self.goal_candidates.append(goal_candidate)
                self.expected_reward.append(candidate_reward + curiosity +
                                            self.restlessness)
                self.time_since_observed.append(0.)
        if len(self.expected_reward) == 0:
            #print 'no expected reward. bailing.'
            return None

        # Estimate the present reward value of each candidate
        decayed_reward = (np.array(self.expected_reward)).ravel() / ( 
                1. + np.array(self.time_since_observed))
        reward_value = decayed_reward - current_reward
        # Find the most rewarding candidate
        best_goal_index = np.where(reward_value == np.max(reward_value))[0][-1]
        highest_reward_value = reward_value[best_goal_index]
        # Check whether the best candidate is good enough to pick 
        # TODO: check whether the winning goal is ever any but the most recent
        #if highest_reward_value >= 0.:
        if True :
            goal = self.goal_candidates.pop(best_goal_index)
            self.expected_reward.pop(best_goal_index)
            self.time_since_observed.pop(best_goal_index)
            self.restlessness = 0.
            self.goal_candidates = []
            self.expected_reward = []
            self.time_since_observed = []

        # If the list of candidates is too long, reduce it
        if len(self.goal_candidates) > self.STM_LENGTH:
            worst_goal_index = np.where(self.expected_reward == 
                                        min(self.expected_reward))[0][0]
            self.goal_candidates.pop(worst_goal_index)
            self.expected_reward.pop(worst_goal_index)
            self.time_since_observed.pop(worst_goal_index)
            
        # debug prints
        #print '--ak'
        #print 'gc', goal_candidate, 'hr', hub_reward, 'c', curiosity
        #print 'canr', candidate_reward, 'curr', current_reward
        return goal

    def visualize(self):
        pass
