"""
One-dimensional grid delay task

This task tests an agent's ability to properly ascribe reward to the 
correct cause. The reward is delayed by a variable amount, which 
makes the task challenging.
"""
import numpy as np
import core.tools as tools
from worlds.base_world import World as BaseWorld
from worlds.grid_1D import World as Grid_1D_World


class World(Grid_1D_World):
    """ 
    One-dimensional grid task with delayed reward
    
    This task is identical to the grid_1D task with the
    exception that reward is randomly delayed a few time steps.
    """
    def __init__(self, lifespan=None, test=False):
        """
        Initialize the world. Base it on the grid_1D world.
        """
        Grid_1D_World.__init__(self, lifespan)
        self.name = 'grid_1D_delay'
        self.name_long = 'one dimensional grid world with delay'
        print '--delayed'
        self.MAX_DELAY = 4
        self.future_reward = [0.] * self.MAX_DELAY
        self.world_visualize_period = 1e6
    
    def assign_reward(self, sensors):
        """
        Calcuate the reward corresponding to the current state and assign
        it to a future time step.
        """
        new_reward = -sensors[8]
        new_reward += sensors[3]
        # Punish actions just a little
        new_reward -= self.energy  * self.ENERGY_COST
        # Find the delay for the reward
        delay = np.random.randint(1, self.MAX_DELAY)
        self.future_reward[delay] = tools.bounded_sum(
                [self.future_reward[delay], new_reward])
        # Advance the reward future by one time step
        self.future_reward.append(0.)
        reward = self.future_reward.pop(0)
        return reward
        
    def visualize_world(self):
        state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
        state_image[self.simple_state] = 'O'
        state_image[self.num_sensors:self.num_sensors + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            for i in range(action_index.size):
                state_image[self.num_sensors + 2 + action_index[i]] = 'x'
        print(''.join(state_image))
