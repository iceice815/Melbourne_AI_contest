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
import math

from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
from game import Actions
import game
from util import nearestPoint
import copy


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """s
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
        self.opponents = self.getOpponents(gameState)
        self.numOfFood = 0
        self.food_positions = self.getFoodYouAreDefending(gameState).asList()
        self.walls = gameState.getWalls().asList()
        self.delay = 0.1
        self.isPowered = 0

        self.food_positions_copy = copy.copy(self.food_positions)
        self.defend_flag = False
        self.defend_count = 0
        self.defend_GPS = []
        self.initial_flag = True
        self.mid_point = random.choice(self.find_mid_region())
        self.step_length = int(math.sqrt(math.pow(gameState.data.layout.width,2)+math.pow(gameState.data.layout.height,2))/6)+1
        print ("step_length:",self.step_length)
        #print self.mid_point

    def find_mid_region(self):
        walls = copy.copy(self.walls)
        x_walls = []
        y_walls = []
        patrol_region = []
        for wall in walls:
            x_walls.append(wall[0])
            y_walls.append(wall[1])
        mid_x_wall = (max(x_walls) - 1) / 2
        mid_y_wall = (max(y_walls) - 1 ) / 2
        if self.red:
            for i in range(mid_x_wall-3, mid_x_wall-1 ):
                for j in range(mid_y_wall-2, mid_y_wall +2):
                    patrol_region.append((i, j))
        else:
            for i in range(mid_x_wall+1, mid_x_wall + 3 ):
                for j in range(mid_y_wall-2, mid_y_wall+2):
                    patrol_region.append((i, j))
        return filter(lambda x: (x not in walls), patrol_region)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.debugClear()
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        #print bestActions

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # time.sleep(self.delay)
        # s = raw_input('press any key to continue:')
        execAction = random.choice(bestActions)

        successor = self.getSuccessor(gameState, execAction)
        myPos = successor.getAgentState(self.index).getPosition()

        if myPos in self.getFood(gameState).asList():
            self.numOfFood += 1
        if self.red:
            if myPos[0] < gameState.data.layout.width / 2:
                self.numOfFood = 0
        else:
            if myPos[0] >= gameState.data.layout.width / 2:
                self.numOfFood = 0

        if myPos in self.getCapsules(gameState):
            self.isPowered = 40

        if self.isPowered != 0:
            self.isPowered -= 1

        self.debugDraw([myPos], [0.5, 0.0, 0.0], False)
        return execAction

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
            enemies = [current_state.getAgentState(
                i) for i in self.getOpponents(current_state)]
            defenders = [a for a in enemies if not a.isPacman and a.getPosition(
            ) != None and a.scaredTimer <= 0]
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

        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()

        #print action, features * weights, features, myPos

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
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        tmp_food = self.numOfFood
        tmp_enemy = None
        # print self.getFood(gameState).asList()
        # print myPos
        if myPos in self.getFood(gameState).asList():
            tmp_food += 1

        # if self.red:

        # else:

            # features['numOfFood'] = self.numOfFood

        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # if myPos[0] >= gameState.data.layout.width / 2 + 1:
        #     tmp_food = 0

        nearest = 6
        # if gameState.getAgentState(self.index).isPacman:

        for opponent in self.opponents:
            # print 'emeny'
            if(gameState.getAgentPosition(opponent) != None and not gameState.getAgentState(opponent).isPacman):
                tmp_dis = self.getMazeDistance(
                    myPos, gameState.getAgentPosition(opponent))
                if nearest > tmp_dis:
                    nearest = tmp_dis
                    tmp_enemy = gameState.getAgentPosition(opponent)

        if nearest == 5:
            features['warning'] = 1
        elif nearest == 4:
            features['warning'] = 2
        elif nearest == 3:
            features['warning'] = 3
        elif nearest == 2:
            features['warning'] = 4
        elif nearest == 1:
            features['warning'] = 10
        elif nearest == 0:
            features['warning'] = 10
            # features['warning'] = 10 / (nearest + 0.01)

        if myPos[1] > gameState.data.layout.width / 2 - 1:
            features['force'] = myPos[1] - \
                (gameState.data.layout.width / 2 - 1)

        if not gameState.getAgentState(self.index).isPacman:
            # features['warning'] = features['warning'] * 5
            # features['forceOffence'] = min(self.getMazeDistance(myPos, p) for p in [(
            #     gameState.data.layout.width / 2 - 1, y) for y in range(0, gameState.data.layout.height)
            #     if not gameState.hasWall(gameState.data.layout.width / 2 - 1, y)])
            if tmp_enemy != None:
                features['vertical'] = self.getMazeDistance(myPos, tmp_enemy)
        if self.isPowered != 0:
            features['warning'] = 0

        #print tmp_food
        if self.numOfFood > 0:
            if self.red:
                features['back'] = abs(tmp_food * - min(self.getMazeDistance(myPos, p) for p in [(
                    gameState.data.layout.width / 2 - 1, y) for y in range(0, gameState.data.layout.height)
                    if not gameState.hasWall(gameState.data.layout.width / 2 - 1, y)]))
            else:
                features['back'] = abs(tmp_food * - min(self.getMazeDistance(myPos, p) for p in [(
                    gameState.data.layout.width / 2, y) for y in range(0, gameState.data.layout.height)
                    if not gameState.hasWall(gameState.data.layout.width / 2, y)]))
            # features['back'] = self.numOfFood * (99 - min(self.getMazeDistance(myPos, p) for p in [(
            #     gameState.data.layout.width / 2 + 1, y) for y in range(0, gameState.data.layout.height)
            #     if not gameState.hasWall(gameState.data.layout.width / 2 + 1, y)]))

        # if self.numOfFood > 0 and features['warning'] != 0:

        #     features['back'] = self.numOfFood * (99 - min(self.getMazeDistance(myPos, p) for p in [(
        #         gameState.data.layout.width / 2 + 1, y) for y in range(0, gameState.data.layout.height)
        #         if not gameState.hasWall(gameState.data.layout.width / 2 + 1, y)]))
            # print features['back']

        if nearest == 6:
            features['distanceToFood'] = 2 * features['distanceToFood']
        else:
            features['back'] = 3 * features['back']

        if(action == Directions.STOP):
            features['stop'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 2000, 'distanceToFood': -40,
                'stop': -5000, 'numOfFood': 100, 'back': -60,
                'warning': -1500, 'forceOffence': -100, 'vertical': 200}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def chooseAction(self, gameState):
        # print "111"
        print self.defendRoaming(gameState)
        print self.aStarSearch(self.defendRoaming(gameState))
        # print "222"
        return self.aStarSearch(self.defendRoaming(gameState))

    def aStarSearch(self, goals):
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)

        for goal in goals:
            dists = util.manhattanDistance(cur_position, goal)
            priorityQueue = util.PriorityQueue()
            priorityQueue.push((cur_position, []), dists)
            visited = []

            while (priorityQueue.isEmpty() == False):
                current_position, traveled = priorityQueue.pop()
                if current_position in visited:
                    continue
                visited.append(current_position)
                if current_position == goal:
                    try:
                        return traveled[0]
                    except:
                        pass
                current_succesors = self.getSuccessors(
                    current_position, currentState)
                for succesor in current_succesors:
                    dists = util.manhattanDistance(succesor[0], goal)
                    cost = len(traveled + [succesor[1]]) + dists
                    priorityQueue.push(
                        (succesor[0], traveled + [succesor[1]]), cost)

        random_position = random.choice(self.food_positions)
        if goals[0] != random_position:
            self.aStarSearch([random_position])
        return 'Stop'

    def getSuccessors(self, current_position, current_state):
        successors = []
        current_walls = copy.copy(self.walls)
        if current_state.getAgentState(self.index).isPacman:
            enemies = [current_state.getAgentState(
                i) for i in self.getOpponents(current_state)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition(
            ) != None and a.scaredTimer <= 0]
            if len(invaders) > 0:
                invaders_position = [i.getPosition() for i in invaders]
                current_walls.extend(invaders_position)
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = current_position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in current_walls:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors



    def find_patrol_region(self):
        walls = copy.copy(self.walls)
        #print walls
        x_walls = []
        y_walls = []
        patrol_region = []
        for wall in walls:
            x_walls.append(wall[0])
            y_walls.append(wall[1])
        mid_x_wall = (max(x_walls) - 1) / 2
        if self.red:
            for i in range(mid_x_wall-3, mid_x_wall-1 ):
                for j in range(1, max(y_walls)):
                    patrol_region.append((i, j))
        else:
            for i in range(mid_x_wall+1, mid_x_wall + 3):
                for j in range(1, max(y_walls)):
                    patrol_region.append((i, j))
        return filter(lambda x: (x not in walls), patrol_region)


    def defendRoaming(self, gameState):

        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        # get invaders position
        enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()!= None]

        if self.initial_flag == False:
            if len(invaders) > 0:
                dists = [self.getMazeDistance(
                    cur_position, a.getPosition()) for a in invaders]
                for i in range(0, len(invaders)):
                    if self.getMazeDistance(cur_position, invaders[i].getPosition()) == min(dists):
                        return [invaders[i].getPosition()]
            elif self.defend_flag == True:
                if (self.defend_count > self.step_length or cur_position == self.defend_GPS[0]):
                    self.defend_flag = False
                    self.defend_count = 0
                self.defend_count = self.defend_count + 1
                return self.defend_GPS
            else:
                foods_positions = self.food_positions_copy
                foods_list = self.getFoodYouAreDefending(gameState).asList()
                if (len(foods_positions) > len(foods_list)):
                    for food in foods_list:
                        try:
                            foods_positions.remove(food)
                        except:
                            pass
                    if (len(foods_positions) > 0):
                        self.food_positions_copy = foods_list
                        self.defend_flag = True
                        self.defend_GPS = foods_positions
                        return self.defend_GPS
                else:
                    patrol_region = self.find_patrol_region()
                    print patrol_region
                    try:
                        patrol_region.remove(currentState)
                    except:
                        pass
                    randomPosition = random.choice(patrol_region)
                    return [randomPosition]
        else:
            if cur_position == self.mid_point :
                self.initial_flag = False
            #print mid_location
            return [self.mid_point]




class StopAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        return 'Stop'
