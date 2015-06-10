"""
benchmark 0.6.2

A suite of worlds to characterize the performance of BECCA variants.
Other agents may use this benchmark as well, as long as they have the 
same interface. (See BECCA documentation for a detailed specification.)
In order to facilitate apples-to-apples comparisons between agents, the 
benchmark will be version numbered.

0.6.2 is a significant change from 0.6.1. It incorporates separate 
training and testing epochs, as is traditional in characterizing 
machine learning algorithms against benchmarks. This has the drawback
of not capturing differences in learning rate. It has the advantages
that it captures ultimate performance levels. This is expecially
appropriate when considering agents that are expected to have
long lifetimes. Ultimate performance is of greater concern than
how fast they climb the initial learning curve.

The length of the training and testing epochs are somewhat arbitrary.
Feel free to adjust them if doing so better captures the ultimate
performance of an agent on the tasks.

Run at the command line as a script with no argmuments:
> python benchmark.py

Becca 0.6.2 performance:
Individual benchmark scores:  [0.8901109781044666, 0.6416460844431342, 0.775659881629856, 0.65019771458443798, 0.76505323762976185, 0.71393510430354068, 0.88913811088287908, 0.74383240353778235, 0.99767270390160268]
Overall benchmark score: 0.785249579891

"""
import matplotlib.pyplot as plt
import numpy as np
import tester
from core.agent import Agent
from worlds.grid_1D import World as World_grid_1D
from worlds.grid_1D_delay import World as World_grid_1D_delay
from worlds.grid_1D_ms import World as World_grid_1D_ms
from worlds.grid_1D_noise import World as World_grid_1D_noise
from worlds.grid_2D import World as World_grid_2D
from worlds.grid_2D_dc import World as World_grid_2D_dc
from worlds.image_1D import World as World_image_1D
from worlds.image_2D import World as World_image_2D
from worlds.fruit import World as World_fruit

def main():
    # Run all the worlds in the benchmark and tabulate their performance
    performance = []
    performance.append(tester.train_and_test(World_grid_1D))
    performance.append(tester.train_and_test(World_grid_1D_delay))
    performance.append(tester.train_and_test(World_grid_1D_ms))
    performance.append(tester.train_and_test(World_grid_1D_noise))
    performance.append(tester.train_and_test(World_grid_2D))
    performance.append(tester.train_and_test(World_grid_2D_dc,
                                             training_period=2e4))
    performance.append(tester.train_and_test(World_image_1D))
    performance.append(tester.train_and_test(World_image_2D,
                                             training_period=2e4))
    performance.append(tester.train_and_test(World_fruit))
    print "Individual benchmark scores: " , performance
    print "Overall benchmark score:", np.mean(np.array(performance)) 
    
    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()
    
if __name__ == '__main__':
    main()
