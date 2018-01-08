import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random
from ple.games.flappybird import FlappyBird
from ple import PLE


def buildModel():
    model = Sequential()
    model.add(Dense(32, input_shape=(3, ), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='nadam', loss='mse')
    return model


def saveModel(model):
    model.save_weights("model.h5", overwrite=True)


def loadModel(model):
    model.load_weights("model.h5")
    model.compile(optimizer='nadam', loss='mse')
    return model


def getMove(model, state, eps):
    if np.random.rand() < eps:
        return (1, eps - 0.001) if np.random.rand() > 0.5 else (0, eps - 0.001)
    return np.argmax(model.predict(state)[0]), eps


def processState(state):
    ans = list()
    ans.append(state["player_vel"])
    ans.append(state["player_y"] - (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) * 0.5)
    ans.append(state["next_pipe_dist_to_player"])
    return np.array(ans).reshape((1, 3))


def trainModel(model, miniBatch):
    gamma = 0.9
    for state, nextMove, reward, nextState, finished in miniBatch:
        finalReward = 0
        if finished:
            finalReward += reward 
        else:
            finalReward += reward + gamma * np.max(model.predict(nextState)[0])
        target = model.predict(state)
        target[0][nextMove] = finalReward
        model.fit(state, target, epochs=1, verbose=False)
    return model


def train(env):
    env.force_fps = True
    moves = [119, None]
    model = buildModel()
#    model = loadModel(buildModel())
    eps = 0.1
    batchSize = 32
    mem = deque(maxlen=10000)
    for epoch in range(1000):
        env.reset_game()
        totalReward, playedGame = 0, 0
        while playedGame < 100:
            state = env.getGameState()
            nextMove, eps = getMove(model, state, eps)
            reward = env.act(moves[nextMove])
            totalReward += reward
            mem.append([state, nextMove, reward, env.getGameState(), env.game_over()])
            if env.game_over():
                print "Epoch %d | Game #%d | Total Reward %.3f" % (epoch, playedGame, totalReward)
                env.reset_game()
                totalReward = 0
                playedGame += 1
                if len(mem) > batchSize:
                    model = trainModel(model, random.sample(mem, batchSize))
                    saveModel(model)
    return model


def play(env, model):
    env.display_screen = True
    env.force_fps = True
    moves = [119, None]
    while True:
        env.reset_game()
        reward = 5
        while not env.game_over():
            state = env.getGameState()
            nextMove, _ = getMove(model, state, 0)
            reward += env.act(moves[nextMove])
        print "Score %d" % reward


if __name__ == '__main__':
    game = FlappyBird()
    env = PLE(game, fps=30, state_preprocessor=processState, display_screen=True, force_fps=False)
#    model = train(env)
    model = loadModel(buildModel())
    play(env, model)