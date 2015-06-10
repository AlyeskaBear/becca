"""
a task in which an robot must choose between ripe and unripe fruit 

In this task, the robot's sensors tell it whether a fruit is either
1) small or large and
2) yellow or purple.
A small yellow fruit is an unripe plum. It is not good to eat.
A small purple fruit is a ripe plum. It is good to eat.
A large yellow fruit is a ripe peach. It is good to eat.
A large purple fruit is a rotten peach. It is not good to eat.

To succeed in this task, the robot has to consider the combination 
of its sensors, rather than each one individually. This is
mathematically related to the XOR task, a challenge for 
many machine learning algorithms.
"""
import numpy as np

import core.tools as tools
from worlds.base_world import World as BaseWorld

class World(BaseWorld):
    """ 
    fruit selection world
    
    In this world, the robot has only two sensors,
    large/small and yellow/purple
    and two actions
    eat or don't eat.
    
    The robot gets rewarded for eating good fruit and 
    punished for eating raw or rotten fruit. This world
    is designed to force the robot to create features 
    from its sensors.
    """
    def __init__(self, lifespan=None, test=False):
        """ 
        Set up the world 
        """
        BaseWorld.__init__(self, lifespan)
        self.name = 'fruit'
        self.name_long = 'fruit selection world'
        print "Entering", self.name_long
        """
        Break out the sensors into
           0: large?
           1: small?
           2: yellow?
           3: purple?
        A sample sensor array would be
            [1., 0., 1., 0.]
        indicating a ripe peach.
        """
        self.num_sensors = 4
        """
        Break out the actions into
            0: eat
            1: discard
            2: do nothing this time step
        """
        self.num_actions = 2
        self.action = np.zeros(self.num_actions)
        self.grab_fruit()
        # If REWARD is 1, the agent is very content and doesn't make 
        # a move on the next time step
        # This might also be due to the fact that the features are built
        # of two-timestep daisychains
        self.REWARD = 1.
        self.verbose =  False

    def grab_fruit(self):
        """
        Grab a new piece of fruit from the box.
        Randomly assign its attributes.
        self.size == 0 # large
        self.size == 1 # small
        self.color == 0 # yellow
        self.color == 1 # purple
        """
        self.size = np.random.randint(2)
        self.color = np.random.randint(2)
        if ( (self.size == 0) and (self.color == 0) or
             (self.size == 1) and (self.color == 1) ):
             self.edible = True
        else: 
            self.edible = False

        self.sensors = np.zeros(self.num_sensors)
        if self.size == 0:
            self.sensors[0] = 1.
        if self.size == 1:
            self.sensors[1] = 1.
        if self.color == 0:
            self.sensors[2] = 1.
        if self.color == 1:
            self.sensors[3] = 1.

    def step(self, action): 
        """ 
        Take one time step through the world 
        """
        self.timestep += 1 
        self.actions = action.ravel()
        self.actions[np.nonzero(self.actions)] = 1.
        self.acted = False
        # Figure out which action was taken
        self.eat = False
        self.discard = False
        if action[0] > .5:
            self.eat = True
            self.acted = True
        elif action[1] > .5:
            self.discard = True
            self.acted = True

        # Check whether the appropriate action was taken, and assign reward.
        # There is a small punishment for doing nothing.
        self.reward = -.1
        if ( (self.eat and self.edible) or 
             (self.discard and not self.edible)):
            self.reward = self.REWARD
        elif ( (self.eat and not self.edible) or 
               (self.discard and self.edible)):
            self.reward = -self.REWARD * .9

        if self.acted:
            self.grab_fruit()

        return self.sensors, self.reward

    def visualize(self, agent=None):
        """ 
        Show what's going on in the world 
        """
        if self.verbose:
            state_str = ' || '.join([ str(self.sensors),
                                      str(self.actions), 
                                      str(self.reward),
                                      str(self.size),
                                      str(self.color),
                                      str(self.timestep) ])
            print state_str
