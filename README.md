Becca is a general learning program for use in any robot or embodied system. When using Becca, a robot learns to do whatever it is rewarded to do, and continues learning throughout its lifetime.

### How do I take a quick look at becca?

    
#### Pull down the code from Pypi.

    pip install becca

`becca_test`, `becca_viz`, and `becca_toolbox` install automatically when you install `becca`. It will also install `numpy`, `numba`, and `matplotlib` if you don't have those in place already.

#### Run it on your local machine.
    
    python3
    >>>import becca_test.test as test
    >>>test.suite()

### How do I install becca for development?

If you want to integrate becca with your robot, simulation, or reinforcement learning benchmark, or you'd like to contribute to the code, you'll need to clone the GitHub repositories and install them locally. Here is [the walkthrough](https://github.com/brohrer/becca/wiki/Installation-walkthrough).

### What can becca do?

Some [videos](http://youtu.be/4kPoU8eZvio?list=PLF861CC4C40439EEB) show it in action. 

### What can becca do for me?

Becca aspires to be a brain for any robot, doing anything. It's not there yet, but it's getting closer.
It may be able to drive your robot. Hook it up and see, using the worlds in the `becca_test` repository
as a model. Feel free to shoot me an email (brohrer@gmail.com) if you'd like to talk it through.

### How does becca 10 work?

I owe you this. It's on my To-Do list.

In the meantime, the reinforcement learner is similar to the one from Becca 7 (described in [this video](https://youtu.be/EXs3nHwLIt0)) and the unsupervised ziptie algorithm hasn't changed from Becca 6 (described on pages 3-6 of [this pdf](https://github.com/brohrer/deprecated-becca-docs/raw/master/how_it_works.pdf)).

The code is also generously documented. I explain all my algorithmic tricks and justify some of my design decisions. I recommend starting at `connector.py` and walking
through from there.

### Next steps.

The good folks at [OpenAI](https://gym.openai.com/) have created a playground called Gym for becca and agents like it.
Learning on simulated robots of all types and complexities is a great opportunity to show what becca can do.
Getting becca integrated with Gym is my next development goal. There are some intermediate steps, and 
I'll be working through them for the next several months.

### Join in

We could use your help! There are [several issues tagged `entrypoint`](https://github.com/brohrer/becca/issues?q=is%3Aissue+is%3Aopen+label%3Aentrypoint). These are a fine place to start if you are coming
to the project for the first time and want to get your feet wet. Tehy aren't necessarily small tasks, or easy ones, but they don't presuppose a deep understanding of the code.

### Questions? Comments? Snide remarks?

Feel free to add or comment on [GitHub issues](https://github.com/brohrer/becca/issues), tag the [becca project on Twitter](https://twitter.com/_brohrer_becca), or send me a personal email (brohrer@gmail.com), as befits the situation. 

<a href="url"><img src="https://github.com/brohrer/becca-docs/raw/master/figs/logo_plate.png" 
align="center" height="40" width="120" ></a>
 

