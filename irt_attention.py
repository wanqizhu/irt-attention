import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import os

GUESS = 0.25
VERSION = 1


# first, let's try predicting ability from difficulty X attention
class StudentAbilityModel(nn.Module):
  def __init__(self, nQuestions = 1000):
    super().__init__() # necessary

    # define any parameters are part of the model
    self.ability = nn.Parameter(torch.zeros(1))
    self.nQuestions = nQuestions
    self.currQuestionCount = 0
    # self.startAttetion = nn.Parameter(torch.zeros(1))
    # self.endAttention = nn.Parameter(torch.zeros(1))


  def forward(self, difficulty, attention):
    prob_correct = torch.sigmoid(self.ability * attention - difficulty)
    return GUESS + (1-GUESS) * prob_correct


# Actually Optimize
def optimize(model, data):

  # this binds the model to the optimizer
  # Notice we set a learning rate (lr)! this is really important
  # in machine learning -- try a few different ones and see what
  # happens to learning.
  optimizer = Adam(model.parameters(), lr=0.005)

  # Pytorch expects inputs and outputs of certain shapes
  # (# data, # features). In our case, we only have 1 feature
  # so the second dimension will be 1. These next two lines
  # transform the data to the right shape!
  difficulty, attention, y = data
  bestParams = None

  # at the beginning, we default minimum loss to infinity
  minLoss = float("inf")

  for i in range(500):
    # wipe any existing gradients from previous iterations!
    # (don't forget to do this for your own code!)
    optimizer.zero_grad()

    # scale attention to be 0/1
    pred = model(difficulty, attention)

    # A loss (or objective) function tells us how "good" a
    # prediction is compared to the true answer
    #
    # This is mathematically equivalent to scoring the truth
    # against a bernoulli distribution with parameters equal
    # to the prediction (a number between 0 and 1)
    loss = F.binary_cross_entropy(pred, y)

    # this step computes all gradients with "autograd"
    # i.e. automatic differentiation
    loss.backward()

    # this actually changes the parameters
    optimizer.step()

    currParams = [model.ability.item()]

    # if the current loss is better than any ones we've seen
    # before, save the parameters.
    if loss.item() < minLoss:
      bestParams = currParams
      minLoss = loss.item()

  output(loss.item(), bestParams, currParams)
  return bestParams


########### Helper methods #########

def output(loss, bestParams, currParams):
  s = 'loss = {:.4f}, ability = {:.4f}'.format(
    loss,
    bestParams[0]
  )
  print(s)


def loadData(prefix='sim'):
  diff = np.genfromtxt(f'data/{prefix}-difficulty-v{VERSION}.csv', delimiter=',')
  diff = torch.from_numpy(diff).float()

  responses = np.genfromtxt(f'data/{prefix}-responses-v{VERSION}.csv', delimiter=',')
  responses = torch.from_numpy(responses).float()
  
  attentions = np.genfromtxt(f'data/{prefix}-attentions-v{VERSION}.csv', delimiter=',')
  attentions = torch.from_numpy(attentions).float()

  return diff, attentions, responses
  


#### generate data ####

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prob_correct(ability, difficulty):
    return GUESS + (1-GUESS) * sigmoid(ability - difficulty)


def run_trial(ability, difficulty):
    p = prob_correct(ability, difficulty)
    rand_trial = np.random.random(size=p.shape)
    result = rand_trial < p

    return result

def gen_data(nQuestions = 30, nStudents = 100):
  '''
  Generate simulated data
  '''

  difficulty = np.random.normal(size=nQuestions)
  abilities = np.random.normal(size=nStudents)

  start_attentions = np.random.random(size=nStudents)
  end_attentions = np.random.random(size=nStudents)

  attentions = [np.linspace(start, end, nQuestions)
      for (start, end) in zip(start_attentions, end_attentions)
  ]

  attentions = np.array(attentions)
  # print(attention)


  np.savetxt(f'data/sim-difficulty-v{VERSION}.csv', difficulty, fmt='%.3f')
  np.savetxt(f'data/sim-abilities-v{VERSION}.csv', abilities, fmt='%.3f')
  np.savetxt(f'data/sim-attentions-v{VERSION}.csv', attentions, delimiter=',', fmt='%.3f')

  attentive_abilities = attentions * abilities.reshape(nStudents, -1)
  #print(attentive_abilities)

  student_answers = run_trial(attentive_abilities, difficulty)
  np.savetxt(f'data/sim-responses-v{VERSION}.csv', student_answers, fmt='%d', delimiter=',')



def main():

  '''
  You are given a simulated dataset of 100 students who took the 
  first section of the GRE math exam (sim-responses.csv).
  In the dataset each student is a row and each question is 
  a column. If element (i, j) is a 1, that means student i correctly 
  answered question j. You know, from testing, the difficulty of 
  each item (sim-difficulty.csv). Infer the ability of each student 
  and save your results as a file (infer-ability.csv). 
  Because the data is simulated, you can compare your estimated abilities
  to the true abilities (sim-abilities.csv).

  '''

  gen_data()
  difficulty, attentions, responses = loadData()
  print(difficulty.size())

  if not os.path.exists(f'data/sim-abilities-pred-v{VERSION}.csv'):
    with open(f'data/sim-abilities-pred-v{VERSION}.csv', 'w') as f:
      for attention, response in zip(attentions, responses):
        model = StudentAbilityModel()
        best_params = optimize(model, (difficulty, attention, response))
        f.write('{:.4f}\n'.format(best_params[0]))

  true_abilities = np.genfromtxt(f'data/sim-abilities-v{VERSION}.csv', delimiter=',')
  pred_abilities = np.genfromtxt(f'data/sim-abilities-pred-v{VERSION}.csv', delimiter=',')
  plt.scatter(true_abilities, pred_abilities)
  plt.plot([-2, 2], [-2, 2])  # plot y=x
  plt.savefig(f'data/sim-abilities-scatter-v{VERSION}.png')
  plt.show()





if __name__ == '__main__':
  main()