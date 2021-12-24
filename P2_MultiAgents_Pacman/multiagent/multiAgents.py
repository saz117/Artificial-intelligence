# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #we focus on eating the dots as long as the ghost is not near pacman

        if successorGameState.isWin():
            return float("inf")

        #we get a list of distance to all the dots, and choose the closet one
        dotsList = []
        for dot in newFood.asList():
            dotsList.append(manhattanDistance(newPos, dot))
        closestFood = min(dotsList)

        #if we dont have the power, and the ghost is too close, we avoid it
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0 and manhattanDistance(newPos, ghost.getPosition()) < 2:
                return float('-inf')
        #the resulting score with respecet to the closest food
        return successorGameState.getScore() + (1.0 / closestFood)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0) #return all legal actions for Pacman
        max_value = float('-inf') #the threshold value so far
        best_action = None  #which we will calculate by the end.

        for action in actions:  #get the max value from all  successors.
            action_value = self.minimax(gameState.generateSuccessor(0, action), 1)
            #now we take the max of all successors/children
            if action_value > max_value:
                max_value = action_value #update the threshold
                best_action = action #update the best action

        return best_action  # return the final action, east, west, stop, etc...

        #we coudl just do it like this, but is not as understandable.
        #return max(actions, key = lambda action: self.minimax(gameState.generateSuccessor(0, action), 1))

    def minimax(self, gameState, turn):
        numAgents = gameState.getNumAgents() #get the current number of players
        agentIndex = turn % numAgents #we can calculate the agentIndex (whos turn is it, by doing turn mod number of agents)
        #so for example if we are on turn 1 (pacman already played) 1 mod 4 = 1, so it's a ghost turn
        # 4 mod 4 = 0 so now its Pacman's turn, and so on
        depth = turn // numAgents #calculate the current depth, at what node are we at?

        if gameState.isLose() or gameState.isWin() or depth == self.depth: #if we win/lose the game, or if we are at a terminal node
            return self.evaluationFunction(gameState) #we return the utility

        actions = gameState.getLegalActions(agentIndex) #get the new actions according to the new agent index
        possibilities = []
        for action in actions:
            possibilities.append(self.minimax(gameState.generateSuccessor(agentIndex, action), turn + 1))

        if agentIndex == 0: #if its pacman
            return max(possibilities) #max because pacman is the player
        else: #if its a ghost
            return min(possibilities) #min because its the opponent

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)  # return all legal actions for Pacman
        alpha, beta = float('-inf'), float('inf')
        best_action = None  # which we will calculate by the end.

        #we use a similar for loop than the one in minimaxAgent, but now we dont use max value, we can use alpha
        #now since it holds the same value as max value did
        for action in actions:  # get the max value from all  successors.
            action_value = self.alphaBeta(gameState.generateSuccessor(0, action), 1, alpha, beta)
            # now we take the max of all successors/children
            if action_value > alpha:
                alpha = action_value  # update the threshold
                best_action = action  # update the best action

        return best_action  # return the final action, east, west, stop, etc...

    def alphaBeta(self, gameState, turn, alpha, beta):
        numAgents = gameState.getNumAgents()
        agentIndex = turn % numAgents
        depth = turn // numAgents

        if gameState.isLose() or gameState.isWin() or depth == self.depth: #if we win/lose the game, or if we are at a terminal node
            return self.evaluationFunction(gameState) #we return the utility
        """
        now the main difference with this method and minimax, is that we cant declare the value variable inside the for action for loop,
        since its now going to change with each successor.
        """
        if agentIndex == 0:  # if its pacman
            value = float('-inf')
        else: #else its a ghost
            value = float('inf')

        actions = gameState.getLegalActions(agentIndex) #get the new actions according to the new agent index
        for action in actions:
            """
            Another big difference is that now we can't just pic the max or min of all possibilities. We now need to 
            iterate over these possibilities and change the alpha and beta values accordingly for each successor. 
            """
            if agentIndex == 0: #if its pacman, we do max-value
                value = max(value, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), turn + 1, alpha, beta))
                if value > beta:
                    return value #prune
                alpha = max(alpha, value)

            else: #if its a ghost, we do min-value
                value = min(value, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), turn + 1, alpha, beta))
                if value < alpha:
                    return value #prune
                beta = min(beta, value)
        #after we change the alpha, beta values accordingly we can now return the value.
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)  # return all legal actions for Pacman
        max_value = float('-inf')  # the threshold value so far
        best_action = None  # which we will calculate by the end.

        for action in actions:  # get the max value from all  successors.
            action_value = self.expectimax(gameState.generateSuccessor(0, action), 1)
            # now we take the max of all successors/children
            if action_value > max_value:
                max_value = action_value  # update the threshold
                best_action = action  # update the best action

        return best_action  # return the final action, east, west, stop, etc..

    def expectimax(self, gameState, turn):
        #we execute a code almost identical to the one used for minimax, but now we calculate the "hope" whenever an opponent plays
        numAgents = gameState.getNumAgents()
        agentIndex = turn % numAgents
        depth = turn // numAgents

        if gameState.isLose() or gameState.isWin() or depth == self.depth:  # if we win/lose the game, or if we are at a terminal node
            return self.evaluationFunction(gameState)  # we return the utility

        actions = gameState.getLegalActions(agentIndex)  # get the new actions according to the new agent index
        possibilities = []
        for action in actions:
            possibilities.append(self.expectimax(gameState.generateSuccessor(agentIndex, action), turn + 1))

        if agentIndex == 0:  # if its pacman
            return max(possibilities)  # max because pacman is the player
        else:  # if its a ghost
            return sum(possibilities) / len(possibilities) #we calculate the hope

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    pacmanPosistion = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    # Now we do a fairly similar method, but instead of saying "avoid this" if pacman is too close to a ghost and scared
    #we can now take the values of the distance and timer to influence the resulting value/score we return

    if currentGameState.isWin():
        return float("inf")

    # we get a list of distance to all the dots, and choose the closet one
    dotsList = []
    for dot in currentFood.asList():
        dotsList.append(manhattanDistance(pacmanPosistion, dot))
    closestFood = min(dotsList)

    #we create a list to store the values of the distance and timer, so we can use it later to calculate how well the value is
    unsafeDistances , scare = [], []
    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPosistion, ghost.getPosition())
        if ghost.scaredTimer == 0: #if we dont have the power
            scare.append(ghost.scaredTimer) #we safe the value of the timer to the list

        if dist < 2: #if the ghost is too close
            unsafeDistances.append(dist) #we save the distance (which is unsafe) to the list

    #now we get the total of these values
    unsafeDistanceValue = sum(unsafeDistances)
    scared = sum(scare)

    #the resulting score with respect to the closest food, proportionaly taking the unsafe distance into account
    # and the total timer value
    return currentGameState.getScore() + (1.0 / closestFood + 1.0 * unsafeDistanceValue) + (1.0/ (scared + 0.01))

# Abbreviation
better = betterEvaluationFunction
