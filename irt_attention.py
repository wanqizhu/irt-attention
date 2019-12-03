import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import os


VERSION = 1.71
GUESS = 0.25
SEED = 1
RESTART = True
nQuestions = 30
nStudents = 100

LOSS_FILE = f'data/v{VERSION}-sim-loss.csv'
PARAMS_FILE = f'data/v{VERSION}-params.json'

DESCRIPTION = {
  1.0 : 'difficulty + attention -> ability, attention random line'
        'with endpoints drawn from unif(0, 1)',
  1.1 : '1.0, but hide attention from prediction model',
  1.2 : '1.0 but doubled attention so its mean is 1, necessary for when we hide attention and set it to be 1',
  1.3 : '1.2, but hide attention from prediction model',
  1.4 : 'attention random walk with mean 1 steps of 0.1 bounded by 0/2',
  1.5 : '1.4 but hide attention',
  1.6 : 'random walk -0.1, 0, 0.1, no clipping',
  1.7 : 'compare 1.6 with a model that just have fixed attention',
  1.71: 'hide attention but model can predict arbitrary attention at each step '
        'see if model can learn this walk',
  1.72: 'model knows the generative function, see if it can retrace the walk '
        'due to the failure of the sign function in computing gradients, '
        'I use scaled sigmoid to approximate each step\'s movement, '
        'the model mostly makes +0.1/-0.1 steps and not much steps close to 0',      
}


# first, let's try predicting ability from difficulty X attention
class StudentAbilityModel(nn.Module):
  def __init__(self, nQuestions = 30):
    super().__init__() # necessary

    # initialize to 2, which is our mean ability
    self.ability = nn.Parameter(2 * torch.ones(1))
    self.nQuestions = nQuestions
    self.currQuestionCount = 0
    self.attention = None 

    if VERSION == 1.71:
      self.attention = nn.Parameter(torch.ones(nQuestions))
    elif VERSION == 1.72:
      self._attention = nn.Parameter(torch.zeros(nQuestions))



  def forward(self, difficulty, attention=None):
    if attention is None:
      attention = 1
    if VERSION == 1.71:
      attention = self.attention
    if VERSION == 1.72:
      # really, we want torch.sign() here, but sign has no gradients
      # so we approximate it to make it centered at 0 and bounded by [-1, 1],
      # mapping positive _attention -> positive deviations that are mostly close to 1
      attn_movements = 2 * torch.sigmoid(100*self._attention) - 1
      attention = 1 + 0.1 * torch.cumsum(attn_movements, dim=0)
      self.attention = attention
      #print(attention)

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
    optimizer.zero_grad()

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

    currParams = [model.ability.item(), model.attention]

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

  if VERSION in [1.71, 1.72]:
    s += ', attention: %s' % bestParams[1]
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

  # FIX: make difficulties and abilities centered around 2 and always
  # nonnegative, o/w attention multiplier behaves unintuitively
  difficulty = 2 + np.random.normal(size=nQuestions)
  difficulty = np.clip(difficulty, 0, 4)
  abilities = 2 + np.random.normal(size=nStudents)
  abilities = np.clip(abilities, 0, 4)

  if VERSION in [1.0, 1.1, 1.2, 1.3]:
    start_attentions = np.random.random(size=nStudents)
    end_attentions = np.random.random(size=nStudents)
    if VERSION in [1.2, 1.3]:  # triple the range of attention variation
      start_attentions *= 2
      end_attentions *= 2

    attentions = [np.linspace(start, end, nQuestions)
        for (start, end) in zip(start_attentions, end_attentions)
    ]

  elif VERSION in [1.4, 1.5]:
    # random walk
    attentions = []
    for i in range(nStudents):
      attn = 1
      attentions.append([])
      for t in range(nQuestions):
        attentions[-1].append(attn)
        attn += 0.1 if np.random.random() > 0.5 else -0.1
        attn = np.clip(attn, 0, 2)

  elif VERSION in [1.6, 1.7, 1.71, 1.72]:
    # random walk no clipping
    attn_movements = np.random.choice([-0.1, 0, 0.1], size=(nStudents, nQuestions))
    attentions = 1 + np.cumsum(attn_movements, axis=1)

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
  gen_data(nQuestions, nStudents)
  difficulty, attentions, responses = loadData()
  print(difficulty.size())

  # run model, save predicetd ability and loss
  # TODO: move this file saving to be configured inside model, like loss
  with open(f'data/v{VERSION}-sim-abilities-pred.csv', 'w') as f:
    global TOTAL_LOSS
    TOTAL_LOSS = 0

    for attention, response in zip(attentions, responses):
      model = StudentAbilityModel(nQuestions)
      if VERSION in [1.0, 1.2, 1.4, 1.6]:
        data = (difficulty, attention, response)
      elif VERSION in [1.1, 1.3, 1.5, 1.7, 1.71, 1.72]:
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
  plt.plot([0, 4], [0, 4])  # plot y=x
  plt.title(f"v{VERSION}: Avg loss (binary_cross_entropy) {avg_loss}")
  plt.xlabel("True Abilities")
  plt.ylabel("Predicted Abilities")
  plt.savefig(f'data/v{VERSION}-sim-abilities-scatter.png')
  plt.show()





if __name__ == '__main__':
  main()