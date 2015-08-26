"""
the base world on which all the other worlds are based
"""
import numpy as np

class World(object):
    """ 
    The base class for creating a new world 
    """
    def __init__(self, lifespan=None):
        """ 
        Initialize a new world with some benign default values 
        """
        if lifespan is None:
            self.LIFESPAN = 10 ** 4
        else:
            self.LIFESPAN = lifespan
        # Starting at -1 allows for an intialization pass.
        self.timestep = -1
        self.world_visualize_period = 1e4
        self.brain_visualize_period = 1e4
        self.name = 'abstract base world'
        # These will likely be overridden in any subclass
        self.num_sensors = 0
        self.num_actions = 0
        self.classifier = False
        
    def step(self, action):
        """ 
        Take a timestep through an empty world that does nothing 
        """
        self.timestep += 1
        sensors = np.zeros(self.num_sensors)
        reward = 0
        return sensors, reward
    
    def is_alive(self):
        """ 
        Returns True when the world has come to an end 
        """
        if(self.timestep < self.LIFESPAN):
            return True
        else:
            return False
   
    def visualize(self, brain):
        """ 
        Let the world show BECCA's internal state as well as its own
        """
        if (self.timestep % self.world_visualize_period) == 0:
            self.visualize_world()
        if (self.timestep % self.brain_visualize_period) == 0:
            brain.visualize()

    def visualize_world(self):
        print('{0} is {1} time steps old.'.format(self.name, self.timestep))

