Becca is a general learning program for use in any robot or embodied system. When using Becca, a robot learns to do whatever it is rewarded to do, and continues learning throughout its lifetime.

### How do I try Becca out?

#### Make sure you have a current version of Anaconda.
    
    conda update conda
    conda update anaconda
    
#### Pull down the code from Pypi.

    pip install becca

`becca_test` installs automatically when you install `becca`. 

#### Run it on your local machine.
    
    python
    >>>import becca_test.test
    >>>becca_test.test.suite()

### What can Becca do?

Some [videos](http://youtu.be/4kPoU8eZvio?list=PLF861CC4C40439EEB) show Becca in action. 

### What can Becca do for me?

Becca aspires to be a brain for any robot, doing anything. It's not there yet, but it's getting closer.
It may be able to drive your robot. Hook it up and see, using the worlds in the `becca_test` repository
as a model. Feel free to shoot me an email (brohrer@gmail.com) if you'd like to talk it through.

### How does Becca 9 work?

I owe you this. It's on my To-Do list.

In the meantime, the reinforcement learner is similar to the one from Becca 7 (described in [this video](https://youtu.be/EXs3nHwLIt0)) and the unsupervised ziptie algorithm hasn't changed from Becca 6 (described on pages 3-6 of [this pdf](https://github.com/brohrer/deprecated-becca-docs/raw/master/how_it_works.pdf)).

The code is also generously documented. I explain all my algorithmic tricks and justify some of my design decisions. I recommend starting at `connector.py` and walking
through from there.

### Next steps.

The good folks at [OpenAI](https://gym.openai.com/) have created a playground called Gym for Becca and agents like it.
Learning on simulated robots of all types and complexities is a great opportunity to show what Becca can do.
Getting Becca integrated with Gym is my next development goal. There are some intermediate steps, and 
I'll be working through them for the next several months.

### Join us

[![Join the chat at https://gitter.im/brohrer/becca](https://badges.gitter.im/brohrer/becca.svg)](https://gitter.im/brohrer/becca?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) provides a forum for users to share questions, solutions, and experiences. 

<a href="url"><img src="https://github.com/brohrer/becca-docs/raw/master/figs/logo_plate.png" 
align="center" height="40" width="120" ></a>
 

