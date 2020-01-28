IT3030 - Deep Learning
----------------------

Solutions for the NTNU course IT3030.

## To do

- Handle hidden layer + 0 hidden layers. Should classification have multiple logits given 0 hidden layers?
- Should the regularization factor be scaled with the number of training examples? And do we need to divide by 2 in addition to having the alpha? Why does the exercise refer to a lambda? Should the bias also have a regularization term?
- When should we dump? After all epochs are finished? In what format should the file be stored? Should we not use NumPy to save?
- Do we need to multiply the L2 loss derivative by 2?
- Do we need to add the regularization to the loss functions and dump it or just handle it in the derivative?
