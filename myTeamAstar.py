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
        self.food_toeat_positions = self.getFood(gameState).asList()
        self.food_positions = self.getFoodYouAreDefending(gameState).asList()
        self.walls = gameState.getWalls().asList()
        self.followed_by_defender = False
        self.need_change = False
        self.another_position = ()

        self.food_positions_copy = copy.copy(self.food_positions)
        self.defend_flag = False
        self.defend_count = 0
        self.defend_GPS = []
        self.initial_flag = True
        self.mid_point = random.choice(self.find_mid_region())
        self.cnt = 0
        self.defendAttack_count = 0

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

    #def chooseAction(self, gameState):

    def find_mid_another(self):
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
                for j in range(mid_y_wall-5, mid_y_wall +5):
                    patrol_region.append((i, j))
        else:
            for i in range(mid_x_wall+1, mid_x_wall + 3 ):
                for j in range(mid_y_wall-5, mid_y_wall+5):
                    patrol_region.append((i, j))
        return filter(lambda x: (x not in walls), patrol_region)

    #def chooseAction(self, gameState):


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

    def aStarSearch(self, goals):
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)

        for goal in goals:
            dists = util.manhattanDistance(cur_position, goal)
            print (dists, goal,cur_position)
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

    def ice_Attack(self,gameState):
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        foods_list = self.getFood(gameState).asList()
        distanceto_foods_list = []
        prio_queue = util.PriorityQueue()
        for food in foods_list:
            dist = self.getMazeDistance(cur_position,food)
            distanceto_foods_list.append(dist)
            prio_queue.push((food),dist)
        foods_sorted_list=[]
        while prio_queue.isEmpty() == False:
            foods_sorted_list.append(prio_queue.pop())
        if len(foods_sorted_list)<4 and len(foods_sorted_list)>0:
            return foods_sorted_list
        elif len(foods_sorted_list)>=4:
            return foods_sorted_list[0:4]


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def chooseAction(self, gameState):
        current_toeat_foods = self.getFood(gameState).asList()
        food_eated = len(self.food_toeat_positions) - len(current_toeat_foods)
        self_currentState = self.getCurrentObservation().getAgentState(self.index)
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)

        # go home
        if len(current_toeat_foods) <= 2:
            print (self.aStarSearch([random.choice(self.food_positions)]),"111")
            return self.aStarSearch([random.choice(self.food_positions)])

        # followed by defender and not pacman
        if self.followed_by_defender == True and not self_currentState.isPacman:
            # need to change place, and random choose place in mid region.
            if self.need_change == False:
                self.need_change = True
                mid_region =self.find_mid_region()
                try:
                    mid_region.remove(cur_position)
                except:
                    pass
                self.another_position = random.choice(mid_region)
                print (self.aStarSearch([self.another_position]),"222")
                return self.aStarSearch([self.another_position])

            else:
            #if not approach the point, continue to go
                if self_currentState.getPosition() !=self.another_position:
                    return self.aStarSearch([self.another_position])
                else:
            #if approach, set do need change and not followed by defender. empty that point
                    self.need_change=False
                    self.followed_by_defender = False
                    self.another_position = ()
                    print "555"

        if self.followed_by_defender == True and self_currentState.isPacman:
            if self.need_change == False:
                self.need_change = True

        #avoid defenders:
        if self_currentState.isPacman:
            print"777"
            # get defenders position
            enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
            defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
            if len(defenders) > 0:
                defenders_position = [i.getPosition() for i in defenders]
                for defender_pos in defenders_position:
                    dist = self.getMazeDistance(defender_pos,self_currentState.getPosition())
                    if dist <=3:
                        print "999"
                        self.followed_by_defender = True
                        print (self.aStarSearch([random.choice(self.food_positions)]),"333")
                        return self.aStarSearch([random.choice(self.food_positions)])
                    print "101010"
            else:
                print "888"
                return self.aStarSearch(self.ice_Attack(gameState))
        # eat random number of food within 5, then go home
        if food_eated >=random.choice([1,2,3,4,5]) and self_currentState.isPacman:
            return self.aStarSearch([random.choice(self.food_positions)])
        if not self_currentState.isPacman:
            self.food_toeat_positions = self.getFood(gameState).asList()
            print (self.aStarSearch(self.ice_Attack(gameState)),"444")
        return self.aStarSearch(self.ice_Attack(gameState))
        print "666"

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def ice_DefenceAttack(self,gameState):
        """
                Picks among the actions with the highest Q(s,a).
                """
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        foods_positions = self.food_positions_copy
        foods_list = self.getFoodYouAreDefending(gameState).asList()
        if(len(foods_positions)>len(foods_list)):
            for food in foods_list:
                try:
                    foods_positions.remove(food)
                except:
                    pass
            if(len(foods_positions)>0):
                self.food_positions_copy = foods_list
                self.defend_flag = True
                self.defend_GPS = foods_positions
                return self.aStarSearch(self.defend_GPS)
        elif self.defend_flag==True:
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

            current_toeat_foods = self.getFood(gameState).asList()
            food_eated = len(self.food_toeat_positions) - len(current_toeat_foods)
            self_currentState = self.getCurrentObservation().getAgentState(self.index)
            currentState = self.getCurrentObservation()
            cur_position = currentState.getAgentPosition(self.index)

            # go home
            if len(current_toeat_foods) <= 2:
                print (self.aStarSearch([random.choice(self.food_positions)]), "111")
                return self.aStarSearch([random.choice(self.food_positions)])

            # followed by defender and not pacman
            if self.followed_by_defender == True and not self_currentState.isPacman:
                # need to change place, and random choose place in mid region.
                if self.need_change == False:
                    self.need_change = True
                    mid_region = self.find_mid_another()
                    try:
                        mid_region.remove(cur_position)
                    except:
                        pass
                    self.another_position = random.choice(mid_region)
                    print (self.aStarSearch([self.another_position]), "222")
                    return self.aStarSearch([self.another_position])

                else:
                    # if not approach the point, continue to go
                    if self_currentState.getPosition() != self.another_position:
                        return self.aStarSearch([self.another_position])
                    else:
                        # if approach, set do need change and not followed by defender. empty that point
                        self.need_change = False
                        self.followed_by_defender = False
                        self.another_position = ()
                        print "555"

            # avoid defenders:
            if self_currentState.isPacman:
                print"777"
                # get defenders position
                enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
                defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
                if len(defenders) > 0:
                    defenders_position = [i.getPosition() for i in defenders]
                    for defender_pos in defenders_position:
                        dist = self.getMazeDistance(defender_pos, self_currentState.getPosition())
                        if dist <= 4:
                            print "999"
                            self.followed_by_defender = True
                            print (self.aStarSearch([random.choice(self.food_positions)]), "333")
                            return self.aStarSearch([random.choice(self.food_positions)])
                        print "101010"
                else:
                    print "888"
                    return self.aStarSearch(self.ice_Attack(gameState))
            # eat random number of food within 5, then go home
            if food_eated >= random.choice([1, 2, 3, 4, 5]) and self_currentState.isPacman:
                return self.aStarSearch([random.choice(self.food_positions)])
            if not self_currentState.isPacman:
                self.food_toeat_positions = self.getFood(gameState).asList()
                print (self.aStarSearch(self.ice_Attack(gameState)), "444")
            return self.aStarSearch(self.ice_Attack(gameState))
            print "666"


    def chooseAction(self, gameState):

        if self.cnt < 55:
            action = self.aStarSearch(self.ice_Defend(gameState))
            #print action
            self.cnt = self.cnt + 1
        else:
            action = self.ice_DefenceAttack(gameState)

        return action



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


    def ice_Defend(self, gameState):

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
                if (self.defend_count > 5 or cur_position == self.defend_GPS[0]):
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
                    #print patrol_region
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
