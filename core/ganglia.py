"""
The Ganglia class
"""
import numpy as np

class Ganglia(object):
    """
    Plan goals and choose actions for the ``brain``.

    The ``Ganglia`` class is analogous to the basal ganglia in the human brain,
    a collection if fascinating structures and circuits deep in one of
    the oldest parts of the brain. No one understands exactly what they
    do but when they are damaged by trauma or disease, people lose 
    the ability to choose and execute actions well. I'm going pretty far out
    on a neuroscientific limb by suggesting that basal ganglionic functions
    include simulation, planning, selection, inhibition, and execution of
    volitional activity, but it is my best guess right now and I'm going
    to run with it.

    Attributes
    ----------
    decay_rate : float
        The fraction by which goal magnitudes are decreased at each 
        time step.
    goals : array of floats
        Goals cover the entire state, both actions and sensors. An action
        goal is an intention to execute that action in the current
        time step. A sensor goal temporarily boosts the reward 
        associated with that sensor. This encourages making decisions
        that activate that sensor.
    num_actions, num_actions : int
        See docstrings for ``brain.py``.    
    """

    def __init__(self, num_sensors, num_actions):
        """
        Configure the ``ganglia``.
        
        Parameters
        ----------
        num_sensors, num_actions : int
            See docstring for ``brain.py``.
        """
        self.num_actions = num_actions
        self.num_sensors = num_sensors
        self.num_elements = self.num_actions + self.num_sensors
        self.goals = np.zeros(self.num_sensors)
        self.decay_rate = .01
        
    def decide(self, features, predicted_actions, decision_scores):
        """
        Choose which, if any, actions to take on this time step.

        This is also an opportunity to prevent predicted_actions
        from the cerebellum from becoming automatic reactions.
        Only one decision can be made per time step. A decision can be
        either to take an action or to set a feature as a goal.

        Parameters
        ----------
        decision_scores : array of floats
            Scores representing the expected reward with taking each 
            action or choosing each feature as a goal.
        features : array of floats
            The current activity of each of the features. 
        predicted_actions : array of floats
            The predicted probability of the actions based on the
            patterns learned by the cerebellum.

        Returns
        -------
        actions : array of floats
            The collection of actions to be executed on the current time
            step.
        goal_index : int
            The index of the feature element chosen as a goal.
        """
        # Decay goals with time.
        self.goals *= 1. - self.decay_rate

        # Let goals be fulfilled by their corresponding features.
        # Only positive goals are considered. Negative goals are 
        # avoidance and those remain in place.
        pos = np.where(self.goals > 0.)
        self.goals[pos] -= features[pos]
        self.goals[pos] = np.maximum(0., self.goals[pos]) 

        # Treat predicted actions as automatic actions.
        decisions = np.zeros(self.num_elements)
        # debug disable predicted actions. These are too high.
        #decisions[:self.num_actions] += predicted_actions
        
        # Choose the decisions with the largest magnitude, whether it's 
        # a reward or a punishment. If it's a reward, choose that 
        # decision, but if it's a punishment, inhibit that decision.
        decision_index = np.argmax(np.abs(decision_scores))
        decision_sign = np.sign(decision_scores[decision_index])
        decisions[decision_index] += decision_sign

        #print 'di',decision_index
        #print 'ds', decision_scores

        # If an inhibition decision was made, then change the 
        # decision index to be the final action, which is always 
        # the 'do nothing' action. This will allow the hippocampus
        # to stil learn appropriately from this time step.
        if decision_sign < 0.:
            #print(' '.join(['Inhibition of index', str(decision_index)]))
            decision_index = self.num_actions - 1
        
        # In the state representation, actions come first.
        actions = decisions[:self.num_actions].copy()
        # Limit actions to the range [0., 1.]
        actions = np.minimum(np.maximum(actions, 0.), 1.)

        # Add the decisions to the ongoing set of goals.
        self.goals += decisions[self.num_actions:]
        # Limit feature goals to the range [-1., 1.]
        # They can be negative because they represent estimated reward.
        self.goals = np.minimum(np.maximum(self.goals, -1.), 1.)

        # Choose a single random action 
        random_action = False
        if random_action:
            decision_index = np.random.randint(self.num_elements)
            decisions[decision_index] = 1. 

        return actions, decision_index
