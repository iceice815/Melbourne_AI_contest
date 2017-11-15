# Team: ZZZTeam
# Member: Derui Wang 679552 deruiw@student.unimelb.edu.au
#         Mark Ting Chun Yong 805780  mting3@student.unimelb.edu.au
#         Shixun Liu 766799 shixunl@student.unimelb.edu.au


# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util, operator
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Agent_one', second='Agent_two'):
    # Attacker
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
# define the control main agant
class ControlAgent(CaptureAgent):
    # define and register intial state
    '''
      Make sure you do not delete the following line. If you would like to
      use Manhattan distances instead of maze distances in order to save
      on initialization time, please take a look at
      CaptureAgent.registerInitialState in captureAgents.py.
  '''

    # initial game
    def registerInitialState(self, gameState):

        # must add this
        CaptureAgent.registerInitialState(self, gameState)

        # all the positions pacman can walk through
        self.available_pos = gameState.getWalls().asList(False)

        # the score pacman currently take
        self.my_score = 0
        self.tmp_my_score = 0
        self.power_mode_steps_left = 0
        self.pacman_type = self.this_pacman_type()  # get pacman type (attacker, defender, others...)
        self.second_role = self.second_role()
        self.attacker_mode = {self.getTeam(gameState)[0]: 'closest_attack',
                              self.getTeam(gameState)[1]: 'furthest_attack'}
        self.delay_attack = 100
        self.tmp_pacman_type = self.pacman_type

        self.mostlikely = [None] * 4

        # Get the grid size and walls
        self.grid_width = gameState.getWalls().width
        self.grid_height = gameState.getWalls().height
        self.walls = gameState.getWalls().asList()

        # the defender will stay around the middle to defend the choke points
        self.chokes = []
        for xOffest in range(1, 4):
            if gameState.isOnRedTeam(self.index):
                x = self.grid_width / 2 - xOffest
            else:
                x = self.grid_width / 2 + xOffest

            for y in range(self.grid_height):
                if not gameState.hasWall(x, y):
                    self.chokes.append((x, y))

        # create beliefs dict for opponent
        global beliefs
        beliefs = [util.Counter()] * gameState.getNumAgents()
        for i, val in enumerate(beliefs):
            if i in self.getOpponents(gameState):
                beliefs[i][gameState.getInitialAgentPosition(i)] = 1.0

    # get successor after run action
    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # check if the position is a dead end
    def deadEnd(self, position, gameState):
        x, y = position
        surround_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        wall_pos = [(x, y) for x, y in surround_positions if gameState.hasWall(x, y)]
        if len(wall_pos) == 3:
            return True
        else:
            return False

    # get the weight of surroundings of position
    def get_weight(self, current_position, gameState):
        x, y = current_position

        possible_positions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        weight_of_next_position = util.Counter()

        for position in possible_positions:
            if position in self.available_pos:
                if self.deadEnd(position, gameState):
                    weight_of_next_position[position] = 0.2
                else:
                    weight_of_next_position[position] = 1
        return weight_of_next_position

    # Looks at how an agent could move from where they currently are
    def get_max_opponent_range(self, gameState):
        for enemy, belief in enumerate(beliefs):
            if enemy in self.getOpponents(gameState):
                new_beliefs = util.Counter()
                # if enemy is visible
                pos = gameState.getAgentPosition(enemy)
                if pos != None:
                    new_beliefs[pos] = 1.0
                else:
                    # Look at all current beliefs
                    for pos in belief:
                        if pos in self.available_pos and belief[pos] > 0:
                            position_weight = self.get_weight(pos, gameState)
                            for x, y in position_weight:  # iterate over these probabilities
                                # expand the range
                                new_beliefs[(x, y)] += belief[pos] * position_weight[(x, y)]
                    # if there is no intersection
                    if len(new_beliefs) == 0:
                        pre_state = self.getPreviousObservation()
                        if pre_state is not None and pre_state.getAgentPosition(enemy) is not None:
                            new_beliefs[pre_state.getInitialAgentPosition(enemy)] = 1.0
                        else:
                            for pos in self.available_pos: new_beliefs[pos] = 1.0
                beliefs[enemy] = new_beliefs

    # get all possible position
    def observe(self, agent, noisy_dist, gameState):
        my_position = gameState.getAgentPosition(self.index)

        # Current state probabilities
        current_probability = util.Counter()
        for position in self.available_pos:  # check each legal position
            true_dist = util.manhattanDistance(position, my_position)  # distance between this point and Pacman
            current_probability[position] = gameState.getDistanceProb(true_dist, noisy_dist)
            # product of old and new probability will outline the max range of opponet
        for pos in self.available_pos:
            beliefs[agent][pos] *= current_probability[pos]

    ######################################## Defender Feature and Weight #########################################


    ######################################
    # Defender: fight with visible enemy #
    ######################################

    def feature_defender_vs_visible_enemy(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get agent current postion
        my_possition = successor.getAgentPosition(self.index)

        # get visible enemies
        enemies = self.getOpponents(successor)
        visible_enemies = [enemy for enemy in enemies if
                           successor.getAgentState(enemy).isPacman and successor.getAgentPosition(enemy) != None]

        features['num_of_visible_enemies'] = len(visible_enemies)

        # distance to closest visible enemies
        if len(visible_enemies) > 0:

            dist_to_closest_visible_enemy = min(
                [self.getMazeDistance(my_possition, successor.getAgentPosition(enemy)) for enemy in visible_enemies])
            features['dist_closest_visible_enemy'] = dist_to_closest_visible_enemy

            # check the thread level of enemy
            if successor.getAgentState(self.index).scaredTimer > 0:
                if dist_to_closest_visible_enemy <= 2:
                    features['thread_level'] = 1
                if dist_to_closest_visible_enemy > 2:
                    features['thread_level'] = -1
                # Dead end heuristic
                actions = successor.getLegalActions(self.index)
                if (len(actions) <= 2):
                    features['deadEnd'] = 1
                else:
                    features['deadEnd'] = 0
            else:
                features['thread_level'] = 0

        if successor.getAgentState(self.index).isPacman:
            features['is_pacman'] = 1
        else:
            features['is_pacman'] = 0

        return features

    def weight_defender_vs_visible_enemy(self):
        return {'num_of_visible_enemies': 1000
            , 'dist_closest_visible_enemy': 50
            , 'deadEnd': 100
            , 'thread_level': 1000
            , 'is_pacman': 1000}

    #############################
    # Defender: defend the gate #
    #############################
    def feature_defender_guard(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        my_possition = successor.getAgentPosition(self.index)

        enemy_pos = None
        closest_dist = float('inf')

        # get the position(accurate or noisy) of opponent
        for enemy in self.getOpponents(successor):
            enemy_position = successor.getAgentPosition(enemy)
            if enemy_position is None:
                enemy_position = self.mostlikely[enemy]
            else:
                dist_to_enemy = self.getMazeDistance(my_possition, enemy_position)
                features['dist_to_enemy'] = dist_to_enemy

            enemy_dist = self.getMazeDistance(my_possition, enemy_position)
            if enemy_dist <= closest_dist:
                closest_dist = enemy_dist
                enemy_pos = enemy_position

        enemy_x, enemy_y = enemy_pos

        # move to the choke that is closest to opponent
        temp = float('inf')
        best_choke = None
        for choke_x, choke_y in self.chokes:
            if abs(choke_y - enemy_y) < temp:
                temp = abs(choke_y - enemy_y)
                best_choke = (choke_x, choke_y)

        dist_to_best_choke = self.getMazeDistance(my_possition, best_choke)
        features['dist_to_best_choke'] = dist_to_best_choke

        # make sure the guard does not move to the opposite
        if successor.getAgentState(self.index).isPacman:
            features['is_pacman'] = 1
        else:
            features['is_pacman'] = 0

        if action == Directions.STOP: features['stop'] = 1

        return features

    def weight_defender_guard(self):
        return {'dist_to_best_choke': 10, 'is_pacman': 1000, 'stop': 1000, 'dist_to_enemy': 50}

    ############################
    # Defender: Predict enemy  #
    ############################
    def feature_track_hidden_enemy(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Get own position
        my_position = successor.getAgentPosition(self.index)

        # Get opponents and invaders
        opponents = self.getOpponents(gameState)
        invaders = [agent for agent in opponents if successor.getAgentState(agent).isPacman]

        # Find number of invaders
        features['num_of_enemies'] = len(invaders)

        # For each invader, calulate its most likely poisiton and distance
        closest_dist = float('inf')
        for enemy in invaders:
            enemy_predict_position = self.mostlikely[enemy]
            enemy_predict_dist = self.getMazeDistance(my_position, enemy_predict_position)
            if enemy_predict_dist <= closest_dist:
                closest_dist = enemy_predict_dist
        features['distance'] = closest_dist

        if successor.getAgentState(self.index).isPacman:
            features['is_pacman'] = 1
        else:
            features['is_pacman'] = 0
        return features

    def weight_track_hidden_enemy(self):

        return {'num_of_enemies': 10, 'distance': 1, 'is_pacman': 1000}

        ######################################## End Defender Feature and Weight ######################################

        ######################################## Attacker Feature and Weight #########################################

        ############################
        # Attacker: Go get food    #
        ############################

    def get_attacker_feature(self, gameState, action):
        attacker_feature = util.Counter()

        # get my curretn position
        successor = self.getSuccessor(gameState, action)
        my_possition = successor.getAgentState(self.index).getPosition()

        # get distance to food
        # get food position
        food_positions = self.getFood(gameState).asList()
        if not food_positions:
            dist_to_closest_food = float('inf')
        else:
            dist_to_closest_food = min(
                [self.getMazeDistance(my_possition, food_position) for food_position in food_positions])
        attacker_feature['dist_to_closest_food'] = dist_to_closest_food

        # left food feature
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        attacker_feature['successorScore'] = len(foodList)

        # dead end featuer
        actions = successor.getLegalActions(self.index)
        if (len(actions) <= 2):
            attacker_feature['deadEnd'] = 1
        else:
            attacker_feature['deadEnd'] = 0

        # both attacker feature
        another_team_member = \
            [another_member for another_member in self.getTeam(gameState) if another_member != self.index][0]

        partner_pos = successor.getAgentPosition(another_team_member)
        if partner_pos is not None:
            dist_to_partner = self.getMazeDistance(my_possition, partner_pos)
            if dist_to_partner == 0:
                dist_to_partner = 0.5
            attacker_feature['dist_to_partner'] = 1.0 / dist_to_partner

        if self.pacman_type == 'Attacker':
            if gameState.getAgentState(another_team_member).isPacman:

                if self.attacker_mode[self.index] == 'closest_attack':
                    dist_to_closest_food = min(
                        [self.getMazeDistance(my_possition, food_position) for food_position in food_positions])
                    attacker_feature['attacker_mode_dist_to_closest_food'] = dist_to_closest_food

                else:
                    dist_to_furthest_food = max(
                        [self.getMazeDistance(my_possition, food_position) for food_position in food_positions])
                    attacker_feature['attacker_mode_dist_to_furthest_food'] = dist_to_furthest_food
                    # when another attack get the food. may affect the other attacker
                    del attacker_feature['successorScore']

        # add random feature to avoid always attacking from the same way
        if action == Directions.STOP: attacker_feature['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: attacker_feature['reverse'] = 1
        return attacker_feature

    def get_attacker_weight(self, gameState):
        attacker_weight = {'successorScore': 500, 'dist_to_closest_food': 500, 'deadEnd': 500, 'stop': 2000,
                           'dist_to_partner': 10000}

        another_team_member = [another_member for another_member in self.getTeam(gameState) if
                               another_member != self.index][0]
        if self.pacman_type == 'Attacker':
            if gameState.getAgentState(another_team_member).isPacman:
                if self.attacker_mode[self.index] == 'closest_attack':
                    attacker_weight = {'successorScore': 500, 'attacker_mode_dist_to_closest_food': 500, 'deadEnd': 500,
                                       'stop': 2000, 'dist_to_partner': 10000}
                else:
                    attacker_weight = {'successorScore': 500, 'attacker_mode_dist_to_furthest_food': 500,
                                       'deadEnd': 500, 'stop': 2000, 'dist_to_partner': 10000}

        for enemy in self.getOpponents(gameState):

            if not gameState.getAgentState(enemy).isPacman:
                if not gameState.getAgentState(enemy).isPacman and gameState.getAgentPosition(enemy) != None:
                    attacker_weight.update({'deadEnd': 1000})

        return attacker_weight

        ########################################
        # Attacker: Get 5 foods then return    #
        ########################################

    def get_five_food_return_feature(self, gameState, my_possition):

        get_five_food_return_feature = util.Counter()

        # let pacman come back to base
        initial_position = gameState.getInitialAgentPosition(self.index)
        dist_to_initial_pos = self.getMazeDistance(my_possition, initial_position)
        get_five_food_return_feature['dist_to_initial_pos'] = dist_to_initial_pos

        return get_five_food_return_feature

    def get_five_food_return_weight(self):
        get_five_food_return_weight = util.Counter()
        get_five_food_return_weight['dist_to_initial_pos'] = 200
        return get_five_food_return_weight

    ########################################
    # Attacker: Afraid of Ghost mechanism  #
    ########################################

    def afraid_of_ghost_feature(self, gameState, my_possition):
        afraid_of_ghost = util.Counter()
        # defend stay away from ghost feature
        for enemy in self.getOpponents(gameState):
            if not gameState.getAgentState(enemy).isPacman:
                enemy_position = gameState.getAgentPosition(enemy)
                if enemy_position == None:
                    afraid_of_ghost['dist_to_enemy'] = 0
                else:
                    dist_to_enemy = self.getMazeDistance(my_possition, enemy_position)
                    if enemy_position == None:
                        afraid_of_ghost['dist_to_enemy'] = 0
                    else:
                        afraid_of_ghost['dist_to_enemy'] = dist_to_enemy

        return afraid_of_ghost

    def afraid_of_ghost_weight(self, gameState, my_next_possition):
        afraid_of_ghost_weight = util.Counter()
        for enemy in self.getOpponents(gameState):
            if not gameState.getAgentState(enemy).isPacman:
                enemy_position = gameState.getAgentPosition(enemy)
                if enemy_position == None or self.getMazeDistance(my_next_possition, enemy_position) > 4:
                    afraid_of_ghost_weight['dist_to_enemy'] = 0
                else:
                    afraid_of_ghost_weight['dist_to_enemy'] = 5000

        return afraid_of_ghost_weight

    ########################################
    # Attacker: Pursure capsule mechanism  #
    ########################################
    def get_capsule_feature(self, gameState, my_possition):
        get_capsule_feature = util.Counter()
        capsule_positions = self.getCapsules(gameState)
        if len(capsule_positions):
            dist_to_closest_capsule = min(
                [self.getMazeDistance(my_possition, capsule_position) for capsule_position in capsule_positions])
            get_capsule_feature['dist_to_closest_capsule'] = dist_to_closest_capsule
        return get_capsule_feature

    def get_capsule_weight(self, gameState, my_next_possition):
        get_capsule_weight = util.Counter()

        capsule_positions = self.getCapsules(gameState)
        if len(capsule_positions):
            dist_to_closest_capsule = min(
                [self.getMazeDistance(my_next_possition, capsule_position) for capsule_position in capsule_positions])

            if dist_to_closest_capsule < 2:
                # 800 go directly
                # put 200 here for testing
                get_capsule_weight['dist_to_closest_capsule'] = -1000
            else:
                # 200 go normal
                get_capsule_weight['dist_to_closest_capsule'] = 0

        else:
            get_capsule_weight['dist_to_closest_capsule'] = 0
        return get_capsule_weight

    ######################################## End Attacker Feature and Weight #####################################
    # evluation includes feature weights
    def evaluate(self, gameState, action):

        ####################### delay attack mechanism#######################
        # check enemy number
        enemy_number = 0
        for enemy in self.getOpponents(gameState):
            if gameState.getAgentState(enemy).isPacman:
                enemy_number += 1
        if enemy_number == 2:
            self.pacman_type = 'destroyer'
        elif enemy_number == 1:
            self.pacman_type = self.second_role
        elif enemy_number == 0:
            if self.pacman_type == 'destroyer':
                self.pacman_type = 'guard'
            if self.pacman_type == 'guard':
                self.delay_attack -= 1
                if self.delay_attack <= 0:
                    self.pacman_type = 'Attacker'
                    self.delay_attack = 100

        ############################ delay attack mechanism ##################


        successor = self.getSuccessor(gameState, action)
        my_possition = gameState.getAgentPosition(self.index)
        my_next_possition = successor.getAgentPosition(self.index)
        score_hold = self.getScore(gameState)
        food_left = self.getFood(gameState).asList()

        # create then eat food mechanism
        if my_next_possition in food_left:
            self.my_score += 1

        # reconstruct my score when agent return our side
        if not gameState.getAgentState(self.index).isPacman:
            self.my_score = 0

        if self.pacman_type == 'guard':
            feature = self.feature_defender_guard(gameState, action)
            weight = self.weight_defender_guard()

        elif self.pacman_type == 'destroyer':  # if agent is defender

            is_enemy_visible = False  # any one is visible, set to True

            for enemy in self.getOpponents(gameState):
                if gameState.getAgentPosition(enemy) is not None:
                    is_enemy_visible = True

            if is_enemy_visible:
                feature = self.feature_defender_vs_visible_enemy(gameState, action)
                weight = self.weight_defender_vs_visible_enemy()
            else:
                feature = self.feature_track_hidden_enemy(gameState, action)
                weight = self.weight_track_hidden_enemy()

        if self.pacman_type == 'Attacker':

            # check if hold 5 score
            if self.my_score >= 3:

                # check power mode move
                if self.power_mode_steps_left > 0 and self.power_mode_steps_left <= 20:
                    feature = self.get_five_food_return_feature(gameState, my_next_possition)
                    weight = self.get_five_food_return_weight()

                    # add afraid of ghost mechanism
                    feature.update(self.afraid_of_ghost_feature(gameState, my_next_possition))
                    weight.update(self.afraid_of_ghost_weight(gameState, my_next_possition))

                feature = self.get_five_food_return_feature(gameState, my_next_possition)
                weight = self.get_five_food_return_weight()

                # add afraid of ghost mechanism
                feature.update(self.afraid_of_ghost_feature(gameState, my_next_possition))
                weight.update(self.afraid_of_ghost_weight(gameState, my_next_possition))

            # less than set winning point, just attack!
            if self.my_score < 4:

                # check if in power mode, start the power mode counter
                if my_next_possition in self.getCapsules(gameState):
                    self.power_mode_steps_left = 41

                # print self.power_mode_steps_left
                # in power mode move
                if self.power_mode_steps_left > 20:

                    self.power_mode_steps_left -= 1
                    # if in power mode. free attack
                    feature = self.get_attacker_feature(gameState, action)
                    weight = self.get_attacker_weight(gameState)

                # used up power mode, normal mode
                else:

                    # check if is pacman, enter opposite field then afraid(sovle ladaju situation)
                    if not gameState.getAgentState(self.index).isPacman:
                        feature = self.get_attacker_feature(gameState, action)
                        weight = self.get_attacker_weight(gameState)

                        feature.update(self.afraid_of_ghost_feature(gameState, my_next_possition))
                        weight.update(self.afraid_of_ghost_weight(gameState, my_next_possition))

                        feature.update(self.get_capsule_feature(gameState, my_next_possition))
                        weight.update(self.get_capsule_weight(gameState, my_next_possition))

                    else:
                        # used up power mode. add afraid ghost and capture capsule mechanism
                        feature = self.get_attacker_feature(gameState, action)
                        weight = self.get_attacker_weight(gameState)

                        feature.update(self.afraid_of_ghost_feature(gameState, my_next_possition))
                        weight.update(self.afraid_of_ghost_weight(gameState, my_next_possition))

                        feature.update(self.get_capsule_feature(gameState, my_next_possition))
                        weight.update(self.get_capsule_weight(gameState, my_next_possition))

        return feature * weight

    def chooseAction(self, gameState):
        # Get all legal actions of current state
        actions = gameState.getLegalActions(self.index)
        # Get list of opponents
        opponents = self.getOpponents(gameState)
        # Get noisey distance data
        noisey_dist = gameState.getAgentDistances()

        # Observe each opponent to get noisey distance measurement and process
        for enemy in opponents:
            self.observe(enemy, noisey_dist[enemy], gameState)

        # Normalise new probabilities and pick most likely location for enemy agent
        for agent in opponents:
            beliefs[agent].normalize()
            self.mostlikely[agent] = max(beliefs[agent].iteritems(), key=operator.itemgetter(1))[0]

        # Do next time step
        self.get_max_opponent_range(gameState)

        values = [(self.evaluate(gameState, action), action) for action in actions]

        next_action = sorted(values, key=lambda x: x[0])[0][1]

        return next_action


# agentone can be defender or offensor
class Agent_one(ControlAgent):
    def this_pacman_type(self):
        return 'Attacker'

    def second_role(self):
        return 'Attacker'


class Agent_two(ControlAgent):
    def this_pacman_type(self):
        return 'Attacker'

    def second_role(self):
        return 'destroyer'