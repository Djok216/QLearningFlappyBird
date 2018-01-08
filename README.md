# QLearningFlappyBird
QLearning Algorithm with Neural Network for FlappyBird.


Graphical interface from https://github.com/ntasfi/PyGame-Learning-Environment


Neural Network structure:
  Input 3 values: 
    1) Bird velocity.
    2) Distance to next pipe.
    3) The difference between my heigh and middle of the pipe gap.
  Two hidden layers with 32 nodes and activation "relu"
  Output layer has 2 values (to flap or not) and activation "linear"
  Optimizer: Nadam

