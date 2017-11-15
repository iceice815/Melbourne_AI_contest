# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import copy

import math

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
from game import Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.food_positions = self.getFoodYouAreDefending(gameState).asList()
    self.walls = gameState.getWalls().asList()
    ## for Go home
    self.randomFoodPos = random.choice(self.getFoodYouAreDefending(gameState).asList())

    self.w = gameState.data.layout.width
    self.h = gameState.data.layout.height
    if self.red:
      return_column = self.w / 2 - 1
    else:
      return_column = self.w / 2
    self.return_points = [(return_column, y) for y in range(self.h) if not gameState.getWalls()[return_column][y]]
    foodLst = copy.deepcopy(self.getFood(gameState).asList())

    # Calculate the heat map

    heatMap = [[0 for x in range(self.h)] for y in range(self.w)]  # [w][h]

    for food in foodLst:
      for x in range(int(self.w / 2), self.w):
        for y in range(self.h):
          if not gameState.getWalls()[x][y]:
            d = self.getMazeDistance(food, (x, y))
            heatMap[x][y] += max(0, 100 - d ** 2)

    localMax = []
    for x in range(int(self.w / 2), self.w):
      for y in range(self.h):
        if not gameState.getWalls()[x][y] and heatMap[x][y] > 0:
          ht = heatMap[x][y]
          if x - 1 >= 0 and not gameState.getWalls()[x - 1][y] and ht < heatMap[x - 1][y]: continue
          if x + 1 < self.w and not gameState.getWalls()[x + 1][y] and ht < heatMap[x + 1][y]: continue
          if y - 1 >= 0 and not gameState.getWalls()[x][y - 1] and ht < heatMap[x][y - 1]: continue
          if y + 1 < self.h and not gameState.getWalls()[x][y + 1] and ht < heatMap[x][y + 1]: continue
          localMax.append(Point([x, y]))
          # self.debugDraw([(x, y)], [0, 1, 0])

    # Remove two farthest food points

    min1 = [0, 0, 999999]  # x, y, heat

    min2 = [0, 0, 999999]  # x, y, heat

    for (x, y) in foodLst:
      if heatMap[x][y] < min1:
        min2 = min1
        min1 = [x, y, heatMap[x][y]]
    foodLst.remove((min1[0], min1[1]))
    foodLst.remove((min2[0], min2[1]))

    # Recalculate the heat map and the local max points

    heatMap = [[0 for x in range(self.h)] for y in range(self.w)]  # [w][h]

    for food in foodLst:
      for x in range(int(self.w / 2), self.w):
        for y in range(self.h):
          if not gameState.getWalls()[x][y]:
            d = self.getMazeDistance(food, (x, y))
            heatMap[x][y] += max(0, 100 - d ** 2)
    localMax = []
    for x in range(int(self.w / 2), self.w):
      for y in range(self.h):
        if not gameState.getWalls()[x][y] and heatMap[x][y] > 0:
          ht = heatMap[x][y]
          if x - 1 >= 0 and not gameState.getWalls()[x - 1][y] and ht < heatMap[x - 1][y]: continue
          if x + 1 < self.w and not gameState.getWalls()[x + 1][y] and ht < heatMap[x + 1][y]: continue
          if y - 1 >= 0 and not gameState.getWalls()[x][y - 1] and ht < heatMap[x][y - 1]: continue
          if y + 1 < self.h and not gameState.getWalls()[x][y + 1] and ht < heatMap[x][y + 1]: continue
          localMax.append(Point([x, y]))
          # self.debugDraw([(x, y)], [0, 1, 0])

    minDiff = 999999
    for _ in range(10):  # try k-means several times to get the best division

      clusters = kmeans(localMax, 2, 0.05)
      localMaxPos = map(lambda p: tuple(p.coords), localMax)
      clusterA = map(lambda p: tuple(p.coords), clusters[0].points)
      clusterB = map(lambda p: tuple(p.coords), clusters[1].points)
      foodClusterA = []
      foodClusterB = []
      for food in foodLst:
        dists = [self.getMazeDistance(food, locMax) for locMax in localMaxPos]
        minDist = min(dists)
        nearestLocalMax = localMaxPos[dists.index(minDist)]
        if nearestLocalMax in clusterA:
          foodClusterA.append(food)
        elif nearestLocalMax in clusterB:
          foodClusterB.append(food)
      diff = abs(len(foodClusterA) - len(foodClusterB))
      if diff < minDiff:
        minDiff = diff
        self.foodClusterA = copy.deepcopy(foodClusterA)
        self.foodClusterB = copy.deepcopy(foodClusterB)

    # To make cluster division deterministic. ClusterA should be on top of clusterB or to the left to it.

    topA = max([food[1] for food in self.foodClusterA])
    topB = max([food[1] for food in self.foodClusterB])
    leftA = min([food[0] for food in self.foodClusterA])
    leftB = min([food[0] for food in self.foodClusterB])
    if topA < topB:
      self.foodClusterA, self.foodClusterB = self.foodClusterB, self.foodClusterA
    elif topA == topB:
      if leftA > leftB:
        self.foodClusterA, self.foodClusterB = self.foodClusterB, self.foodClusterA

    for food in foodLst:
      if food in self.foodClusterA:
        self.debugDraw(food, [1, 0, 0])
      elif food in self.foodClusterB:
        self.debugDraw(food, [0, 0, 1])

    # maxHeat = max(map(max, heatMap))

    # for x in range(w):

    #     for y in range(h):

    #         if not gameState.getWalls()[x][y] and heatMap[x][y] > 0:

    #             self.debugDraw([(x, y)], [heatMap[x][y] * 1.0 / maxHeat, 0, 0])


    self.heatMap = heatMap

  def defendRoaming(self, gameState):

    currentState = self.getCurrentObservation()
    cur_position = currentState.getAgentPosition(self.index)
    # get invaders position
    enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    if len(invaders) > 0:
      dists = [self.getMazeDistance(cur_position, a.getPosition()) for a in invaders]
      for i in range(0, len(invaders)):
        if self.getMazeDistance(cur_position, invaders[i].getPosition()) == min(dists):
          return [invaders[i].getPosition()]

    else:
      foodList = self.getFoodYouAreDefending(gameState).asList()
      foodListX = []
      PrimDefandPosition = []
      # defPosition = random.choice(foodList)

      for x in range(0, len(foodList)): foodListX.append(foodList[x][0])

      sortedList = sorted(foodListX)

      # Understand the Range of the map.
      thisWalls = copy.copy(self.walls)
      xline = []
      yline = []
      for i in thisWalls:
        xline.append(i[0])
      largerX = max(xline)

      for i in thisWalls:
        yline.append(i[1])

      largery = max(yline)
      midwax = (largerX - 1) / 2

      gride = []
      newclearG = []

      for i in range(midwax, largerX):
        for j in range(1, largery):
          gride.append((i, j))

      gride = filter(lambda x: x != thisWalls, gride)

      for i in gride:
        if (midwax + 1) == i[0] or (midwax + 2) == i[0] or (midwax + 3) == i[0]:
          newclearG.append(i)

      randomPosition = random.choice(newclearG)

      if cur_position == randomPosition:
        newclearG.remove(randomPosition)
        randomPosition = random.choice(newclearG)
      return [randomPosition]

  def aStarSearch(self, goals):

    """Search the node that has the lowest combined cost and heuristic first."""
    currentState = self.getCurrentObservation()
    cur_position = currentState.getAgentPosition(self.index)

    for goal in goals:

      prioQueue = util.PriorityQueue()
      prioQueue.push((cur_position, []), util.manhattanDistance(cur_position, goal))
      visitedNodes = []

      while (prioQueue.isEmpty() == False):

        cur_position1, wholeWay = prioQueue.pop()

        if cur_position1 in visitedNodes:
          continue
        visitedNodes.append(cur_position1)

        if cur_position1 == goal:
          return wholeWay[0]
        cur_succ = self.getSuccessors(cur_position1, currentState)
        for s in cur_succ:
          cost = len(wholeWay + [s[1]]) + util.manhattanDistance(s[0], goal)
          prioQueue.push((s[0], wholeWay + [s[1]]), cost)

    # cannot find way then go home
    random_position = random.choice(self.food_positions)
    if goals[0] != random_position:
      self.aStarSearch([random_position])
    # cannot find way go home then wait for die
    return 'Stop'


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getSuccessors(self, current_position, current_state):
    successors = []
    current_walls = copy.copy(self.walls)
    if current_state.getAgentState(self.index).isPacman:
      enemies = [current_state.getAgentState(i) for i in self.getOpponents(current_state)]
      defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
      if len(defenders) > 0:
        defenders_position = [i.getPosition() for i in defenders]
        current_walls.extend(defenders_position)
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x, y = current_position
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if (nextx, nexty) not in current_walls:
        nextState = (nextx, nexty)
        successors.append((nextState, action))
    return successors

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  food_list = None
  heatMap = None

  def registerInitialState(self, gameState):
    ReflexCaptureAgent.registerInitialState(self, gameState)
    self.food_list = self.foodClusterA
    self.heatMap = [[0 for x in range(self.h)] for y in range(self.w)]
    for food in self.food_list:
      for x in range(self.w):
        for y in range(self.h):
          if not gameState.getWalls()[x][y]:
            d = self.getMazeDistance(food, (x, y))
            self.heatMap[x][y] += 2 ** (100 - d)

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    actions.pop(actions.index('Stop'))
    values = [self.evaluate(gameState, action) for action in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    best_action = bestActions[0]
    best_successor = gameState.generateSuccessor(self.index, best_action)
    best_position = best_successor.getAgentPosition(self.index)
    if best_position in self.food_list:
      self.food_list.pop(self.food_list.index(best_position))
      i = 1
      for x in range(self.w):
        for y in range(self.h):
          if not gameState.getWalls()[x][y]:
            # is not wall

            d = self.getMazeDistance((x, y), best_position)
            self.heatMap[x][y] -= 2 ** (100 - d)
    return best_action

  def evaluate(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    position = successor.getAgentPosition(self.index)
    if self.food_list:
      return self.heatMap[position[0]][position[1]]
    else:
      return 100 - min([self.getMazeDistance(return_point, position) for return_point in self.return_points])


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def chooseAction(self, gameState):
    return self.aStarSearch(self.defendRoaming(gameState))

class Point(object):
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)


class Cluster(object):
    def __init__(self, points):
        if len(points) == 0:
            raise Exception("ERROR: empty cluster")
        self.points = points
        self.n = points[0].n
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]
        return Point(centroid_coords)


def kmeans(points, k, cutoff):
    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]
    loopCounter = 0
    while True:
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)
        loopCounter += 1
        for p in points:
            smallest_distance = getDistance(p, clusters[0].centroid)
            clusterIndex = 0
            for i in range(clusterCount - 1):
                distance = getDistance(p, clusters[i + 1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i + 1
            lists[clusterIndex].append(p)
        biggest_shift = 0.0
        for i in range(clusterCount):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)
        if biggest_shift < cutoff:
            break
    return clusters


def getDistance(a, b):
    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i] - b.coords[i]), 2)
        accumulatedDifference += squareDifference
    distance = math.sqrt(accumulatedDifference)
    return distance