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
import random
import util
import math

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        foodPositions = newFood.asList()
        stateEvaluation = 0

        for ghostState in newGhostStates:
            if ghostState.scaredTimer > 2:
                continue
            else:
                distance = manhattanDistance(newPos, ghostState.getPosition())
                if 0 < distance <= 4:
                    stateEvaluation -= (1/distance) * 100000

        if successorGameState.getNumFood() < currentGameState.getNumFood():
            stateEvaluation += 1000
        else:
            stateEvaluation += 1/manhattanDistance(newPos, min(foodPositions))
        return stateEvaluation


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        # start the search with pacman at max depth
        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def minimaxSearch(self, gameState, agentIndex, depth):
        return runSearchProblem(self, gameState, agentIndex, depth)

    def minimizer(self, gameState, agentIndex, depth, legalActions):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        minimum = math.inf
        min_action = None
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.minimaxSearch(successor, next_agent, next_depth)[0]
            if score < minimum:
                minimum, min_action = score, action
        return minimum, min_action

    def maximizer(self, gameState, agentIndex, depth, legalActions):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        maximum = -math.inf
        max_action = None
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.minimaxSearch(successor, next_agent, next_depth)[0]
            if score > maximum:
                maximum, max_action = score, action
        return maximum, max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBetaSearch(gameState, agentIndex=0, depth=self.depth, alpha=-math.inf, beta=math.inf)[1]

    def alphaBetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth, legalActions, alpha, beta)
        else:
            return self.minimizer(gameState, agentIndex, depth, legalActions, alpha, beta)

    def minimizer(self, gameState, agentIndex, depth, legalActions, alpha, beta):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        minimum = math.inf
        min_action = None
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.alphaBetaSearch(successor, next_agent, next_depth, alpha, beta)[0]
            if score < minimum:
                minimum, min_action = score, action
            if score < alpha:
                return score, action
            beta = min(beta, minimum)
        return minimum, min_action

    def maximizer(self, gameState, agentIndex, depth, legalActions, alpha, beta):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        maximum = -math.inf
        max_action = None
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.alphaBetaSearch(successor, next_agent, next_depth, alpha, beta)[0]
            if score > maximum:
                maximum, max_action = score, action
            if score > beta:
                return score, action
            alpha = max(alpha, maximum)
        return maximum, max_action


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
        return self.expectimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def expectimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth, legalActions)
        else:
            return self.expectimax(gameState, agentIndex, depth, legalActions)

    def expectimax(self, gameState, agentIndex, depth, legalActions):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        expectimax_score = 0
        min_action = Directions.STOP
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            expectimax_score += self.expectimaxSearch(successor, next_agent, next_depth)[0]
        return expectimax_score * (1 / len(legalActions)), min_action

    def maximizer(self, gameState, agentIndex, depth, legalActions):
        next_agent, next_depth = getNextAgentAndDepth(gameState, agentIndex, depth)
        maximum = -math.inf
        max_action = Directions.STOP
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.expectimaxSearch(successor, next_agent, next_depth)[0]
            if score > maximum:
                maximum, max_action = score, action
        return maximum, max_action


def getNextAgentAndDepth(gameState, agentIndex, depth):
    # if we have reached the last agent, next agent is pacman, and we decrease depth
    # otherwise we increment the agentIndex and continue at the same depth
    if agentIndex == gameState.getNumAgents() - 1:
        return 0, depth - 1
    else:
        return agentIndex + 1, depth


# runs generic search problem with respect to "self"
def runSearchProblem(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth, legalActions)
        else:
            return self.minimizer(gameState, agentIndex, depth, legalActions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: this evaluation function evaluates the positions of ghosts (and their
    scared status), and whether or not we are moving to a position with new food or not.
    if we are, that is usually preferred.  If not, we move towards the closest pellet.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    evaluation = currentGameState.getScore()

    # ghost checks
    for ghost in newGhostStates:
        if ghost.getPosition() == newPos:
            if ghost.scaredTimer > 1:
                return math.inf
            else:
                return -math.inf

    # food evaluation
    if newPos in newFood.asList():
        evaluation += 10
    elif len(newFood.asList()) > 0:
        evaluation += 1 / min([manhattanDistance(pellet, newPos) for pellet in newFood.asList()])

    return evaluation

# Abbreviation
better = betterEvaluationFunction
