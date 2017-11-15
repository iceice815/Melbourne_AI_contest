"""
Team Name: Pacman RongYao
boaol 748171 boaol@student.unimelb.edu.au
xieb1 741012 xieb1@student.unimelb.edu.au
dhuo1 791094 dhuo1@student.unimelb.edu.au

"""


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
import math


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


class OffensiveReflexAgent(CaptureAgent):
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
        self.stuck = 0
        self.attack = 0
        self.isPowered = 0
        self.plan = []
        self.draw = []
        self.area = ['Top', 'Mid', 'Bottom']

        self.step_length = int(math.sqrt(math.pow(
            gameState.data.layout.width, 2) + math.pow(gameState.data.layout.height, 2)) / 6) + 1
        self.food_positions_copy = copy.copy(self.food_positions)
        self.defend_flag = False
        self.defend_count = 0
        self.defend_GPS = []
        self.initial_flag = True
        self.mid_point = random.choice(self.find_mid_region())
        self.cnt = 0
        self.defendAttack_count = 0
        # print self.mid_point

    def find_mid_region(self):
        walls = copy.copy(self.walls)
        x_walls = []
        y_walls = []
        patrol_region = []
        for wall in walls:
            x_walls.append(wall[0])
            y_walls.append(wall[1])
        mid_x_wall = (max(x_walls) - 1) / 2
        mid_y_wall = (max(y_walls) - 1) / 2
        if self.red:
            for i in range(mid_x_wall - 3, mid_x_wall - 1):
                for j in range(mid_y_wall - 2, mid_y_wall + 2):
                    patrol_region.append((i, j))
        else:
            for i in range(mid_x_wall + 2, mid_x_wall + 4):
                for j in range(mid_y_wall - 2, mid_y_wall + 2):
                    patrol_region.append((i, j))
        return filter(lambda x: (x not in walls), patrol_region)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        myPos = gameState.getAgentPosition(self.index)

        self.debugClear()

        if myPos == gameState.getInitialAgentPosition(self.index):
            self.plan = []
            self.draw = []

        area = []

        if len(self.plan) > 0:
            astaraction = self.plan.pop()
            self.draw.pop()
            self.debugDraw(self.draw, [0.5, 0, 0], True)
            # print self.plan
            # print 'planed action', astaraction

            flag = False
            if self.red:
                if gameState.getAgentPosition(self.index)[0] >= gameState.data.layout.width / 2:
                    self.plan = []
                    self.draw = []
                    self.stuck = 0
                    flag = True

            else:
                if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2:
                    self.plan = []
                    self.draw = []
                    self.stuck = 0
                    flag = True

            if not flag:

                successor = self.getSuccessor(gameState, astaraction)
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
                return astaraction

        nearest = 6

        for opponent in self.opponents:
            if gameState.getAgentPosition(opponent) != None and not gameState.getAgentState(
                    opponent).isPacman and gameState.getAgentState(self.index).isPacman:
                tmp_dis = self.getMazeDistance(
                    myPos, gameState.getAgentPosition(opponent))
                if nearest > tmp_dis:
                    nearest = tmp_dis

        if nearest < 6 and self.isPowered < 10:

            escape, state, pos = self.aStarSearchEscape(
                gameState, self.getNearstback(gameState))
            escape.reverse()

            self.debugDraw(pos, [0.5, 0, 0], True)

            execaction = escape.pop()
            successor = self.getSuccessor(gameState, execaction)
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

            return execaction

        if self.red:
            if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2 - 4:
                self.plan = []
                self.draw = []
                self.stuck = 0
        else:
            if gameState.getAgentPosition(self.index)[0] > gameState.data.layout.width / 2 + 3:
                self.plan = []
                self.draw = []
                self.stuck = 0

        if self.red:
            if gameState.getAgentPosition(self.index)[0] > gameState.data.layout.width / 2 - 3:
                self.attack = 1
        else:
            if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2 + 2:
                self.attack = 1
        if self.red:
            if self.attack == 1:
                if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2:
                    self.stuck += 1
                else:
                    self.stuck = 0
        else:
            if self.attack == 1:
                if gameState.getAgentPosition(self.index)[0] >= gameState.data.layout.width / 2:
                    self.stuck += 1
                else:
                    self.stuck = 0

        if self.stuck > 5:
            area = []
            if myPos[1] in range(0, gameState.data.layout.height / 3):
                area = ['Mid', 'Top']
            elif myPos[1] in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3):
                area = ['Bottom', 'Top']
            elif myPos[1] in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height + 1):
                area = ['Mid', 'Bottom']

            anotherArea = random.choice(area)

            if self.red:
                if anotherArea == 'Bottom':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                            for y in range(0, gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                elif anotherArea == 'Mid':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                            for y in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3) if
                            not gameState.hasWall(x, y)]
                elif anotherArea == 'Top':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                            for y in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height) if
                            not gameState.hasWall(x, y)]
            else:
                if anotherArea == 'Bottom':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                            for y in range(0, gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                elif anotherArea == 'Mid':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                            for y in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3) if
                            not gameState.hasWall(x, y)]
                elif anotherArea == 'Top':
                    area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                            for y in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height) if
                            not gameState.hasWall(x, y)]

            target = random.choice(area)

            self.plan, state, self.draw = self.aStarSearchPlan(
                gameState, target)
            self.debugDraw(self.draw, [0.5, 0, 0], True)

            self.plan.reverse()
            self.draw.reverse()

            self.stuck = 0

            astaraction = self.plan.pop()

            successor = self.getSuccessor(gameState, astaraction)
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

            return astaraction
            # You can profile your evaluation time by uncommenting these lines
            # start = time.time()

        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # print bestActions

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

        return features * weights

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

        if myPos in self.getCapsules(gameState):
            features['powered'] = 1

        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        if len(self.getCapsules(successor)) > 0:
            if features['getCapsules'] == -1:
                features['getCapsules'] = 10
            if features['getCapsules'] == -1:
                features['getCapsules'] = 5
            if features['getCapsules'] == -2:
                features['getCapsules'] = 4
            if features['getCapsules'] == -3:
                features['getCapsules'] = 3
            if features['getCapsules'] == -4:
                features['getCapsules'] = 2
            if features['getCapsules'] == -5:
                features['getCapsules'] = 1

        nearest = 6

        for opponent in self.opponents:
            # print 'emeny'
            if (gameState.getAgentPosition(opponent) != None and not gameState.getAgentState(opponent).isPacman):
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

        # print tmp_food
        if self.numOfFood > 0:
            if self.red:
                features['back'] = abs(tmp_food * - min(self.getMazeDistance(myPos, p) for p in [(
                    gameState.data.layout.width / 2 - 1,
                    y) for y in
                    range(0,
                          gameState.data.layout.height)
                    if
                    not gameState.hasWall(
                    gameState.data.layout.width / 2 - 1,
                    y)]))
            else:
                features['back'] = abs(tmp_food * - min(self.getMazeDistance(myPos, p) for p in [(
                    gameState.data.layout.width / 2,
                    y) for y in
                    range(0,
                          gameState.data.layout.height)
                    if
                    not gameState.hasWall(
                    gameState.data.layout.width / 2,
                    y)]))
        if self.isPowered > 5:
            features['warning'] = 0
            features['back'] = features['back'] / 3

        if (action == Directions.STOP):
            features['stop'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 2000, 'distanceToFood': -90,
                'stop': -5000, 'numOfFood': 100, 'back': -60,
                'warning': -1500, 'powered': 5000, 'getCapsules': 400}

    def getNearstback(self, gameState):
        myPos = gameState.getAgentPosition(self.index)

        back = []
        if self.red:
            back = [(gameState.data.layout.width / 2 - 1, y) for y in range(0, gameState.data.layout.height)
                    if not gameState.hasWall(gameState.data.layout.width / 2 - 1, y)]
        else:
            back = [(gameState.data.layout.width / 2, y) for y in range(0, gameState.data.layout.height)
                    if not gameState.hasWall(gameState.data.layout.width / 2, y)]
        nearst = 99999
        pointPos = None
        for point in back:
            tmp = self.getMazeDistance(myPos, point)
            if nearst > tmp:
                nearst = tmp
                pointPos = point
        return pointPos

    def aStarSearchPlan(self, gameState, target):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE ***"

        myPos = gameState.getAgentPosition(self.index)
        pq = util.PriorityQueue()
        foodList = self.getFood(gameState).asList()
        currentState = gameState
        explored = []
        plan = [[], currentState, []]

        while myPos != target:
            if myPos in explored:
                plan = pq.pop()
                myPos = plan[1].getAgentPosition(self.index)
                continue
            else:
                explored.append(myPos)

            for action in plan[1].getLegalActions(self.index):
                successor = self.getSuccessor(plan[1], action)
                nextPos = successor.getAgentPosition(self.index)
                if nextPos not in explored:
                    tmp = copy.deepcopy(plan)
                    tmp[0].append(action)
                    tmp[1] = successor
                    tmp[2].append(nextPos)
                    pq.push(tmp, util.manhattanDistance(myPos, target))

            plan = pq.pop()

            myPos = plan[1].getAgentPosition(self.index)

        # print "cost, ", len(plan[0])

        return plan

    def aStarSearchEscape(self, gameState, target):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE ***"

        myPos = gameState.getAgentPosition(self.index)
        pq = util.PriorityQueue()
        foodList = self.getFood(gameState).asList()
        currentState = gameState
        explored = []
        plan = [[], currentState, []]

        def escape(plan):
            nearest = 6

            for opponent in self.opponents:
                # print 'emeny'
                if plan[1].getAgentPosition(opponent) != None and not (plan[1].getAgentState(opponent).isPacman):
                    tmp_dis = self.getMazeDistance(
                        plan[2][-1], plan[1].getAgentPosition(opponent))
                    if nearest > tmp_dis:
                        nearest = tmp_dis
            if nearest == 6:
                return -6
            else:
                return -nearest

        while myPos != target:
            if myPos in explored:
                plan = pq.pop()
                myPos = plan[1].getAgentPosition(self.index)
                continue
            else:
                explored.append(myPos)

            for action in plan[1].getLegalActions(self.index):
                successor = self.getSuccessor(plan[1], action)
                nextPos = successor.getAgentPosition(self.index)
                for opponent in self.opponents:
                    # print 'emeny'
                    if successor.getAgentPosition(opponent) != None and not (
                            successor.getAgentState(opponent).isPacman):
                        if nextPos == successor.getAgentPosition(opponent):
                            explored.append(nextPos)
                if nextPos not in explored:
                    tmp = copy.deepcopy(plan)
                    tmp[0].append(action)
                    tmp[1] = successor
                    tmp[2].append(nextPos)
                    pq.push(tmp, util.manhattanDistance(nextPos, target))

            plan = pq.pop()

            myPos = plan[1].getAgentPosition(self.index)

        return plan


class DefensiveReflexAgent(OffensiveReflexAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def ice_DefenceAttack(self, gameState):
        """
                Picks among the actions with the highest Q(s,a).
                """
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        enemies = [currentState.getAgentState(
            i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]

        foods_positions = self.food_positions_copy
        foods_list = self.getFoodYouAreDefending(gameState).asList()
        if(len(foods_positions) > len(foods_list)):
            for food in foods_list:
                try:
                    foods_positions.remove(food)
                except:
                    pass
            if(len(foods_positions) > 0):
                self.food_positions_copy = foods_list
                self.defend_flag = True
                self.defend_GPS = foods_positions
                self.plan = []
                self.draw = []
                return self.aStarSearch(self.defend_GPS)
        elif self.defend_flag == True:
            if len(invaders) > 0:
                dists = [self.getMazeDistance(
                    cur_position, a.getPosition()) for a in invaders]
                for i in range(0, len(invaders)):
                    if self.getMazeDistance(cur_position, invaders[i].getPosition()) == min(dists):
                        return self.aStarSearch([invaders[i].getPosition()])
            elif (self.defendAttack_count > 20 or cur_position == self.defend_GPS[0]):
                self.defend_flag = False
                self.defendAttack_count = 0
            self.defendAttack_count = self.defendAttack_count + 1
            return self.aStarSearch(self.defend_GPS)
        else:

            """
            Picks among the actions with the highest Q(s,a).
            """
            actions = gameState.getLegalActions(self.index)

            myPos = gameState.getAgentPosition(self.index)

            self.debugClear()

            if myPos == gameState.getInitialAgentPosition(self.index):
                self.plan = []
                self.draw = []

            area = []

            if len(self.plan) > 0:
                astaraction = self.plan.pop()
                self.draw.pop()
                self.debugDraw(self.draw, [0.5, 0, 0], True)
                print self.plan
                print 'planed action', astaraction

                flag = False
                if self.red:
                    if gameState.getAgentPosition(self.index)[0] >= gameState.data.layout.width / 2:
                        self.plan = []
                        self.draw = []
                        self.stuck = 0
                        flag = True

                else:
                    if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2:
                        self.plan = []
                        self.draw = []
                        self.stuck = 0
                        flag = True

                if not flag:

                    successor = self.getSuccessor(gameState, astaraction)
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
                    return astaraction

            nearest = 6
            # if gameState.getAgentState(self.index).isPacman:

            for opponent in self.opponents:
                # print 'emeny'
                if gameState.getAgentPosition(opponent) != None and not gameState.getAgentState(opponent).isPacman and gameState.getAgentState(self.index).isPacman:
                    tmp_dis = self.getMazeDistance(
                        myPos, gameState.getAgentPosition(opponent))
                    if nearest > tmp_dis:
                        nearest = tmp_dis

            if nearest < 6 and self.isPowered < 10:

                escape, state, pos = self.aStarSearchEscape(
                    gameState, self.getNearstback(gameState))
                escape.reverse()
                print 'escape', escape
                self.debugDraw(pos, [0.5, 0, 0], True)

                execaction = escape.pop()
                successor = self.getSuccessor(gameState, execaction)
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

                return execaction

            if self.red:
                if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2 - 4:
                    self.plan = []
                    self.draw = []
                    self.stuck = 0
            else:
                if gameState.getAgentPosition(self.index)[0] > gameState.data.layout.width / 2 + 3:
                    self.plan = []
                    self.draw = []
                    self.stuck = 0

            if self.red:
                if gameState.getAgentPosition(self.index)[0] > gameState.data.layout.width / 2 - 3:
                    self.attack = 1
            else:
                if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2 + 2:
                    self.attack = 1
            if self.red:
                if self.attack == 1:
                    if gameState.getAgentPosition(self.index)[0] < gameState.data.layout.width / 2:
                        self.stuck += 1
                    else:
                        self.stuck = 0
            else:
                if self.attack == 1:
                    if gameState.getAgentPosition(self.index)[0] >= gameState.data.layout.width / 2:
                        self.stuck += 1
                    else:
                        self.stuck = 0

            if self.stuck > 5:
                print 'stuck'
                area = []
                if myPos[1] in range(0, gameState.data.layout.height / 3):
                    area = ['Mid', 'Top']
                elif myPos[1] in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3):
                    area = ['Bottom', 'Top']
                elif myPos[1] in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height + 1):
                    area = ['Mid', 'Bottom']

                print 'area ', area
                anotherArea = random.choice(area)
                print 'another', anotherArea

                if self.red:
                    if anotherArea == 'Bottom':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                                for y in range(0, gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                    elif anotherArea == 'Mid':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                                for y in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                    elif anotherArea == 'Top':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2 - 3, gameState.data.layout.width / 2)
                                for y in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height) if not gameState.hasWall(x, y)]
                else:
                    if anotherArea == 'Bottom':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                                for y in range(0, gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                    elif anotherArea == 'Mid':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                                for y in range(gameState.data.layout.height / 3, 2 * gameState.data.layout.height / 3) if not gameState.hasWall(x, y)]
                    elif anotherArea == 'Top':
                        area = [(x, y) for x in range(gameState.data.layout.width / 2, gameState.data.layout.width / 2 + 3)
                                for y in range(2 * gameState.data.layout.height / 3, gameState.data.layout.height) if not gameState.hasWall(x, y)]

                print 'area ..', area
                target = random.choice(area)
                print 'target', target
                self.plan, state, self.draw = self.aStarSearch2(
                    gameState, target)
                self.debugDraw(self.draw, [0.5, 0, 0], True)

                self.plan.reverse()
                self.draw.reverse()
                print 'plan0000', self.plan
                self.stuck = 0

                astaraction = self.plan.pop()
                print 'planed action - first', astaraction

                successor = self.getSuccessor(gameState, astaraction)
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

                return astaraction
                # You can profile your evaluation time by uncommenting these lines
                # start = time.time()

            values = [self.evaluate(gameState, a) for a in actions]
            # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]

            print bestActions

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
    # for defend attack
    # def chooseAction(self, gameState):
    #
    #     if self.cnt < 50:
    #         action = self.aStarSearch(self.ice_Defend(gameState))
    #         # print action
    #         self.cnt = self.cnt + 1
    #     else:
    #         action = self.ice_DefenceAttack(gameState)
    #
    #     return action

    # for defend only
    def chooseAction(self, gameState):
        return self.aStarSearch(self.ice_Defend(gameState))

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
        x_walls = []
        y_walls = []
        patrol_region = []
        for wall in walls:
            x_walls.append(wall[0])
            y_walls.append(wall[1])
        mid_x_wall = (max(x_walls) - 1) / 2
        if self.red:
            for i in range(mid_x_wall - 3, mid_x_wall - 1):
                for j in range(1, max(y_walls)):
                    patrol_region.append((i, j))
        else:
            for i in range(mid_x_wall + 2, mid_x_wall + 4):
                for j in range(1, max(y_walls)):
                    patrol_region.append((i, j))
        return filter(lambda x: (x not in walls), patrol_region)

    def ice_Defend(self, gameState):

        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        enemies = [currentState.getAgentState(
            i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]

        if self.initial_flag == False:
            if len(invaders) > 0:
                dists = [self.getMazeDistance(
                    cur_position, a.getPosition()) for a in invaders]
                for i in range(0, len(invaders)):
                    if self.getMazeDistance(cur_position, invaders[i].getPosition()) == min(dists):
                        return [invaders[i].getPosition()]
            elif self.defend_flag == True:
                if (self.defend_count > self.step_length - 1 or cur_position == self.defend_GPS[0]):
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
                    # print patrol_region
                    try:
                        patrol_region.remove(currentState)
                    except:
                        pass
                    randomPosition = random.choice(patrol_region)
                    return [randomPosition]
        else:
            if cur_position == self.mid_point:
                self.initial_flag = False
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
