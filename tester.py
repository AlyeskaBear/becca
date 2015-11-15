{\rtf1\ansi\ansicpg1252\cocoartf1348\cocoasubrtf170
{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
Connect a BECCA brain to a world and run them.\
\
To use this module as a top level script: \
    1. Select the World that the Brain will be placed in.\
    2. Make sure the appropriate import line is included and \
        uncommented below. \
    3. Run from the command line.\
        > python tester.py\
"""\
\
import cProfile\
import numpy as np\
import pstats\
\
# Worlds from the benchmark\
#from worlds.base_world import World\
#from worlds.grid_1D import World\
#from worlds.grid_1D_delay import World\
#from worlds.grid_1D_ms import World\
#from worlds.grid_1D_noise import World\
#from worlds.grid_2D import World\
#from worlds.grid_2D_dc import World\
from worlds.image_1D import World\
#from worlds.image_2D import World\
#from worlds.fruit import World\
\
# If you want to run a world of your own, add the appropriate line here\
#from worlds.hello import World\
#from becca_world_chase_ball.chase import World\
#from becca_world_chase_ball.simple_chase import World\
#from becca_world_mnist.mnist import World\
#from becca_world_watch.watch import World\
#from becca_world_listen.listen import World\
\
from core.brain import Brain \
\
def train_and_test(world_class, training_period=1e4, testing_period=1e4):\
    """\
    First train the brain on a world, then test the brain's performance. \
    \
    Parameters\
    ----------\
    world_class : World\
        The class containing the BECCA-compatible world that the \
        brain will be receiving sensor and reward information from and \
        sending action commands to.\
    training_period, testing_period : int, optional\
        The number of time steps to train or test the brain\
        on the current world.  The default is 10,000.\
\
    Returns\
    -------\
    test_performance : float\
        The average reward per time step during the testing period.\
    """\
    # Train the brain on the world\
    world = world_class(lifespan=training_period)\
    train_average = run(world)\
    train_performance = train_average * training_period \
    world = world_class(lifespan=testing_period, test=True)\
    # Test the brain on the world\
    total_average = run(world, restore=True, exploit=True)\
    total_performance = total_average * (training_period + testing_period)\
    test_performance = ((total_performance - train_performance) / \
                        testing_period )\
\
    print('Test performance is: \{0:.3\}'.format(test_performance))\
    return test_performance\
\
def run(world, restore=False, exploit=False):\
    """ \
    Run BECCA with a world. \
\
    Connects the brain and the world together and runs them for as long \
    as the world dictates.\
\
    Parameters\
    ----------\
    restore : bool, optional\
        If ``restore`` is True, try to restore the brain from a previously saved\
        version, picking up where it left off. \
        Otherwise it create a new one. The default is False.\
    exploit : bool, optional\
        If ``exploit`` is True, \
        The default is False.\
\
    Returns\
    -------\
    performance : float\
        The performance of the brain over its lifespan, measured by the\
        average reward it gathered per time step.\
    """\
    brain_name = '\{0\}_brain'.format(world.name)\
    brain = Brain(world.num_sensors, world.num_actions, \
                  brain_name=brain_name, exploit=exploit)\
    if restore:\
        brain = brain.restore()\
    # Start at a resting state.\
    actions = np.zeros(world.num_actions)\
    sensors, reward = world.step(actions)\
    # Repeat the loop through the duration of the existence of the world:\
    # sense, act, repeat.\
    while(world.is_alive()):\
        actions = brain.sense_act_learn(sensors, reward)\
        sensors, reward = world.step(actions)\
        world.visualize(brain)\
    performance = brain.report_performance()\
    return performance\
\
def profile():\
    """\
    Profile the brain's performance on the selected world.\
    """\
    profiling_lifespan = 1e4\
    print('Profiling BECCA\\'s performance...')\
    command = 'run(World(lifespan=\{0\}), restore=True)'.format(\
            profiling_lifespan)\
    cProfile.run(command, 'tester.profile')\
    p = pstats.Stats('tester.profile')\
    p.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)\
    print('   View at the command line with')\
    print(' > python -m pstats tester.profile')\
    \
if __name__ == '__main__':\
    # To profile BECCA's performance with world, set profile_flag to True.\
    profile_flag = False\
    if profile_flag:\
        profile()\
    else:\
        run(World(lifespan=1e8), restore=True)\
}