import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import os


VERSION = 1.0
GUESS = 0.25
SEED = 1
RESTART = True

LOSS_FILE = f'data/v{VERSION}-sim-loss.csv'
PARAMS_FILE = f'data/v{VERSION}-params.json'

DESCRIPTION = {
  1.0 : 'difficulty + attention -> ability, attention random line'
        'with endpoints drawn from unif(0, 1)',
  1.1 : '1.0, but hide attention from prediction model',
  1.2 : '1.0 but tripled attention',
  1.3 : '1.2, but hide attention from prediction model',
}


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


  def forward(self, difficulty, attention=None):
    if attention is None:
      attention = 1
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
  if len(data) == 3:
    difficulty, attention, y = data
  else:
    difficulty, y = data
    attention = None

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

  with open(LOSS_FILE, 'a') as f:
    output(loss.item(), bestParams, currParams, f)
  return bestParams


########### Helper methods #########

TOTAL_LOSS = 0
def output(loss, bestParams, currParams, file=None):
  global TOTAL_LOSS
  TOTAL_LOSS += loss

  if file is not None:
    file.write('{:.4f}\n'.format(loss))
  
  s = 'loss = {:.4f}, ability = {:.4f}'.format(
    loss,
    bestParams[0]
  )
  print(s)


def loadData():
  diff = np.genfromtxt(f'data/v{VERSION}-sim-difficulty.csv', delimiter=',')
  diff = torch.from_numpy(diff).float()

  responses = np.genfromtxt(f'data/v{VERSION}-sim-responses.csv', delimiter=',')
  responses = torch.from_numpy(responses).float()
  
  attentions = np.genfromtxt(f'data/v{VERSION}-sim-attentions.csv', delimiter=',')
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
  if VERSION in [1.2, 1.3]:  # triple the range of attention variation
    start_attentions *= 3
    end_attentions *= 3

  attentions = [np.linspace(start, end, nQuestions)
      for (start, end) in zip(start_attentions, end_attentions)
  ]

  attentions = np.array(attentions)
  # print(attention)


  np.savetxt(f'data/v{VERSION}-sim-difficulty.csv', difficulty, fmt='%.3f')
  np.savetxt(f'data/v{VERSION}-sim-abilities.csv', abilities, fmt='%.3f')
  np.savetxt(f'data/v{VERSION}-sim-attentions.csv', attentions, delimiter=',', fmt='%.3f')

  attentive_abilities = attentions * abilities.reshape(nStudents, -1)
  #print(attentive_abilities)

  student_answers = run_trial(attentive_abilities, difficulty)
  np.savetxt(f'data/v{VERSION}-sim-responses.csv', student_answers, fmt='%d', delimiter=',')



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
  if not RESTART and os.path.exists(f'data/v{VERSION}-sim-abilities-pred.csv'):
    return

  # reset output files
  if RESTART:
    if os.path.exists(LOSS_FILE):
      os.remove(LOSS_FILE)

  # log params
  with open(PARAMS_FILE, 'w') as f:
    f.write(f'{{version: {VERSION}, guess: {GUESS}, seed: {SEED},'
            f' description: {DESCRIPTION[VERSION]}}}\n')


  np.random.seed(SEED)

  # generate the simulated data
  gen_data()
  difficulty, attentions, responses = loadData()
  print(difficulty.size())

  # run model, save predicetd ability and loss
  # TODO: move this file saving to be configured inside model, like loss
  with open(f'data/v{VERSION}-sim-abilities-pred.csv', 'w') as f:
    global TOTAL_LOSS
    TOTAL_LOSS = 0

    for attention, response in zip(attentions, responses):
      model = StudentAbilityModel()
      if VERSION in [1.0, 1.2]:
        data = (difficulty, attention, response)
      elif VERSION in [1.1, 1.3]:
        data = (difficulty, response)
      else:
        raise ValueError(f"Unknown version: {VERSION}")

      best_params = optimize(model, data)
      f.write('{:.4f}\n'.format(best_params[0]))

  avg_loss = TOTAL_LOSS / len(responses)
  print('Avg loss: ', avg_loss)

  true_abilities = np.genfromtxt(f'data/v{VERSION}-sim-abilities.csv', delimiter=',')
  pred_abilities = np.genfromtxt(f'data/v{VERSION}-sim-abilities-pred.csv', delimiter=',')
  plt.scatter(true_abilities, pred_abilities)
  plt.plot([-2, 2], [-2, 2])  # plot y=x
  plt.title(f"v{VERSION}: Avg loss (binary_cross_entropy) {avg_loss}")
  plt.xlabel("True Abilities")
  plt.ylabel("Predicted Abilities")
  plt.savefig(f'data/v{VERSION}-sim-abilities-scatter.png')
  plt.show()





if __name__ == '__main__':
  main()