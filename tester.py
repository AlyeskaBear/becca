"""
Connect a BECCA agent to a world and run them.

To use this module as a top level script, select the World that the Agent 
will be placed in.
Make sure the appropriate import line is included and uncommented below. 
Run from the command line.
> python tester.py
"""

import cProfile
import numpy as np
import pstats

# Worlds from the benchmark
#from worlds.base_world import World
#from worlds.grid_1D import World
#from worlds.grid_1D_delay import World
#from worlds.grid_1D_ms import World
#from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World
#from worlds.fruit import World
# If you want to run a world of your own, add the appropriate line here
#from worlds.hello import World
#from becca_world_chase_ball.chase import World
#from becca_world_chase_ball.simple_chase import World
from becca_world_mnist.mnist import World
#from becca_world_watch.watch import World

from core.agent import Agent 

def train_and_test(world_module, training_period=1e4, testing_period=1e4):
    """
    First train the agent on a world, then test the agent's performance. 
    
    Return the test performance, the average reward per timestep 
    during the testing period.
    """
    world = world_module(lifespan=training_period)
    train_average = run(world, show=False)
    train_performance = train_average * training_period 
    world = world_module(lifespan=testing_period, test=True)
    total_average = run(world, restore=True, exploit=True, show=False)
    total_performance = total_average * (training_period + testing_period)
    test_performance = ((total_performance - train_performance) / 
                        testing_period )
    print 'Test performance is:', test_performance
    return test_performance

def run(world, restore=False, exploit=False, show=True):
#def run(world, restore=False, exploit=True, show=True):
    """ 
    Run BECCA with a world 

    If restore is True, this method loads a saved agent if it can find one.
    Otherwise it creates a new one. It connects the agent and
    the world together and runs them for as long as the 
    world dictates.
    """
    agent_name = '_'.join((world.name, 'agent'))
    agent = Agent(world.num_sensors, world.num_actions, 
                  agent_name=agent_name, exploit=exploit, 
                  classifier=world.classifier, show=show)
    if restore:
        agent = agent.restore()
    actions = np.zeros((world.num_actions,1))
    # Repeat the loop through the duration of the existence of the world 
    while(world.is_alive()):
        sensors, reward = world.step(actions)
        world.visualize(agent)
        actions = agent.step(sensors, reward)
    return agent.report_performance()

def profile():
    """
    Profile the agent's performance on the selected world.
    """
    profiling_lifespan = 1e3
    print 'profiling BECCA\'s performance...'
    #cProfile.run('run(World(lifespan=profiling_lifespan), restore=True)', 
    #             'tester_profile')
    cProfile.run(''.join(['run(World(lifespan=', str(profiling_lifespan),
                          '), restore=True)']), 'tester.profile')
    p = pstats.Stats('tester.profile')
    p.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)
    print '   View at the command line with' 
    print ' > python -m pstats tester.profile'
    
if __name__ == '__main__':
    # To profile BECCA's performance with world, set profile_flag to True.
    profile_flag = False
    if profile_flag:
        profile()
    else:
        run(World(lifespan=1e8), restore=True)
