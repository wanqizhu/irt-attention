import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import os


VERSION = 1.72
GUESS = 0.25
SEED = 1
RESTART = True
nQuestions = 30
nStudents = 100

LOSS_FILE = f'data/v{VERSION}-sim-loss.csv'
PARAMS_FILE = f'data/v{VERSION}-params.json'
REAL_RESPONSES_FILE = 'math_question_answers.txt'
REAL_DIFFICULTIES_FILE = 'math_question_difficulties.txt'

DESCRIPTION = {
  1.0 : 'difficulty + attention -> ability, attention random line'
        'with endpoints drawn from unif(0, 1)',
  1.1 : '1.0, but hide attention from prediction model',
  1.2 : '1.0 but doubled attention to be ~unif(0, 2) so its mean is 1,' 
        ' necessary for when we hide attention and set it to be 1',
  1.3 : '1.2, but hide attention from prediction model',
  1.4 : 'attention random walk with mean 1 steps of 0.1 bounded by 0/2',
  1.5 : '1.4 but hide attention',
  1.6 : 'random walk -0.1, 0, 0.1, no clipping',
  1.7 : '1.6 with a model that just have fixed attention',
  1.71: 'hide attention but model can predict arbitrary attention at each step '
        ' see if model can learn this walk',
  1.72: 'model knows the generative function, see if it can retrace the walk '
        ' due to the failure of the sign function in computing gradients, '
        ' I use scaled sigmoid to approximate each step\'s movement, '
        ' the model mostly makes +0.1/-0.1 steps and not much steps close to 0',      
  1.73: '1.72 but the model prediction must be bounded by [0, 2]',


  2.0 : 'real data',
  2.72 : '+model also can vary attention in up to 0.1 intervals',
}


# first, let's try predicting ability from difficulty X attention
class StudentAbilityModel(nn.Module):
  def __init__(self, nQuestions = 30, initAbility = 2):
    super().__init__() # necessary

    # initialize to mean ability
    self.ability = nn.Parameter(initAbility * torch.ones(1))
    self.nQuestions = nQuestions
    self.initAbility = initAbility
    self.currQuestionCount = 0
    self.attention = None 

    if VERSION == 1.71:
      self.attention = nn.Parameter(torch.ones(nQuestions))
    elif VERSION in [1.72, 1.73, 2.72, 2.73]:
      self._attention = nn.Parameter(torch.zeros(nQuestions))



  def forward(self, difficulty, attention=None):
    if attention is None:
      attention = 1
    if VERSION == 1.71:
      attention = self.attention
    if VERSION in [1.72, 1.73, 2.72, 2.73]:
      # really, we want torch.sign() here, but sign has no gradients
      # so we approximate it to make it centered at 0 and bounded by [-1, 1],
      # mapping positive _attention -> positive deviations that are mostly close to 1
      attn_movements = 2 * torch.sigmoid(100*self._attention) - 1
      attention = 1 + 0.1 * torch.cumsum(attn_movements, dim=0)
      if VERSION in [1.73, 2.73]:
        attention = torch.clamp(attention, 0, 2)
      self.attention = attention

    prob_correct = torch.sigmoid(self.ability * attention - difficulty)
    return GUESS + (1-GUESS) * prob_correct


# Actually Optimize
def optimize(model, data):

  # this binds the model to the optimizer
  # Notice we set a learning rate (lr)! this is really important
  # in machine learning -- try a few different ones and see what
  # happens to learning.
  optimizer = Adam(model.parameters(), lr=0.0025 * model.initAbility)

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

  if VERSION in [1.71, 1.72, 1.73]:
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

def gen_sim_data(nQuestions = 30, nStudents = 100):

  # make difficulties and abilities centered around 2 and always
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

  elif VERSION in [1.6, 1.7, 1.71, 1.72, 1.73]:
    # random walk no clipping
    attn_movements = np.random.choice([-0.1, 0, 0.1], 
                                      size=(nStudents, nQuestions))
    attentions = 1 + np.cumsum(attn_movements, axis=1)

    plt.figure()
    plt.title(f'v{VERSION} generated attentions')
    for attn in attentions:
      plt.plot(attn)
    plt.savefig(f'data/v{VERSION}-generated-attentions-plot.jpg')
      
    if VERSION == 1.6: plt.show()

  attentions = np.array(attentions)
  # print(attention)

  np.savetxt(f'data/v{VERSION}-sim-difficulty.csv', difficulty, fmt='%.3f')
  np.savetxt(f'data/v{VERSION}-sim-abilities.csv', abilities, fmt='%.3f')
  np.savetxt(f'data/v{VERSION}-sim-attentions.csv', attentions, delimiter=',', fmt='%.3f')

  attentive_abilities = attentions * abilities.reshape(nStudents, -1)
  #print(attentive_abilities)

  student_answers = run_trial(attentive_abilities, difficulty)
  np.savetxt(f'data/v{VERSION}-sim-responses.csv', student_answers, fmt='%d', delimiter=',')


def setup():
  if not RESTART and os.path.exists(f'data/v{VERSION}-sim-abilities-pred.csv'):
    return False

  # reset output files
  if RESTART:
    if os.path.exists(LOSS_FILE):
      os.remove(LOSS_FILE)

  # log params
  with open(PARAMS_FILE, 'w') as f:
    f.write(f'{{version: {VERSION}, guess: {GUESS}, seed: {SEED},'
            f' description: {DESCRIPTION[VERSION]}}}\n')

  np.random.seed(SEED)

  global TOTAL_LOSS
  TOTAL_LOSS = 0
  return True


def main():
  if not setup(): return

  if VERSION < 2:
    # generate the simulated data
    gen_sim_data(nQuestions, nStudents)
    difficulty, attentions, responses = loadData()
    print(difficulty.size())


    attentions_pred = []
    # run model, save predicetd ability and loss
    # TODO: move this file saving to be configured inside model, like loss
    with open(f'data/v{VERSION}-sim-abilities-pred.csv', 'w') as f:

      for attention, response in zip(attentions, responses):
        model = StudentAbilityModel(nQuestions)
        if VERSION in [1.0, 1.2, 1.4, 1.6]:
          data = (difficulty, attention, response)
        elif VERSION in [1.1, 1.3, 1.5, 1.7, 1.71, 1.72, 1.73]:
          data = (difficulty, response)
        else:
          raise ValueError(f"Unknown version: {VERSION}")

        best_params = optimize(model, data)
        f.write('{:.4f}\n'.format(best_params[0]))
        if best_params[1] is not None:
          attentions_pred.append(best_params[1].tolist())

    avg_loss = TOTAL_LOSS / len(responses)
    print('Avg loss: ', avg_loss)

    true_abilities = np.genfromtxt(f'data/v{VERSION}-sim-abilities.csv', delimiter=',')
    pred_abilities = np.genfromtxt(f'data/v{VERSION}-sim-abilities-pred.csv', delimiter=',')
    plt.figure()
    plt.scatter(true_abilities, pred_abilities)
    corr = np.corrcoef(true_abilities, pred_abilities)[0][1]  # Pearson Correlation
    err_squared = np.mean((true_abilities - pred_abilities)**2)
    plt.plot([0, 4], [0, 4])  # plot y=x
    plt.title(f"v{VERSION}: response loss {avg_loss:.2f}, ability corr {corr:.2f}, err^2 {err_squared:.2f}")
    plt.xlabel("True Abilities")
    plt.ylabel("Predicted Abilities")
    plt.savefig(f'data/v{VERSION}-sim-abilities-scatter.png')
    plt.show()

  elif VERSION >= 2.0:
    # use real data
    responses = open(REAL_RESPONSES_FILE, 'r').readlines()
    difficulties = open(REAL_DIFFICULTIES_FILE, 'r').readlines()

    attentions_pred = []
    perct_correct = []
    with open(f'data/v{VERSION}-sim-abilities-pred.csv', 'w') as f:
      with open(f'data/v{VERSION}-sim-attentions-pred.csv', 'w') as g:
        for res, dif in zip(responses, difficulties):
          res = list(map(int, res.strip().split(', ')))
          res = torch.tensor(res).float()
          dif = list(map(int, dif.strip().split(', ')))
          dif = torch.tensor(dif).float()
          assert len(res) == len(dif)

          model = StudentAbilityModel(nQuestions=len(res), initAbility = 4)
          data = (dif, res)

          best_params = optimize(model, data)
          f.write('{:.4f}\n'.format(best_params[0]))
          if best_params[1] is not None:
            # write attention
            attention = best_params[1].tolist()
            g.write(','.join(map(str, attention)) + '\n')
            attentions_pred.append(attention)

          perct_correct.append(res.sum().item() / len(res) * 100)

    avg_loss = TOTAL_LOSS / len(responses)
    print('Avg loss: ', avg_loss)

    # validating that our predictions make some sense
    pred_abilities = np.genfromtxt(f'data/v{VERSION}-sim-abilities-pred.csv', delimiter=',')
    plt.figure()
    plt.scatter(pred_abilities, perct_correct)
    plt.title(f"v{VERSION}: response loss: {avg_loss:.2f}")
    plt.savefig(f'data/v{VERSION}-sim-abilities-scatter.png')
    plt.show()

  # plot attentions, if any
  if attentions_pred:
    plt.figure()
    for attn in attentions_pred:
      plt.plot(attn)

    plt.title(f'v{VERSION} predicted attentions')
    plt.savefig(f'data/v{VERSION}-sim-attentions-plot.jpg')
    plt.show()


if __name__ == '__main__':
  main()