# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import random
import os
import json
import numpy as np

from Game import *


# =========================
# Inference bot (LEGACY style)
# =========================

class Bot(object):
    """
    Legacy AIVoid style:
      - output is sigmoid in [0,1] as "risk/cost" (lower is better)
      - choose action by MIN cost
    """

    def __init__(self, Theta1, Theta2, game):
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.game = game

    def Sigmoid(self, x):
        # stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def PreProcess(self, action):
        state_new = []
        for aster in self.game.asteroids:
            state_new.append(aster[0] / (self.game.Halfwidth + 0.0))
            state_new.append(aster[1] / (self.game.Height + 0.0))
        state_new.append(1)  # bias

        # symmetry trick: action L flips x
        if action == 'L':
            for i in range(self.game.N):
                state_new[2 * i] *= -1

        layer1 = np.empty([2 * self.game.N + 1, 1])
        for i in range(2 * self.game.N + 1):
            layer1[i, 0] = state_new[i]
        return layer1   #Obtain the normalized coordinates +bias

    def ForwardPropagate(self, action):
        layer1 = self.PreProcess(action)
        layer2_temp = np.dot(np.transpose(self.Theta1), layer1)
        for i in range(layer2_temp.shape[0]):
            layer2_temp[i, 0] = self.Sigmoid(layer2_temp[i, 0])
        layer2 = np.append(layer2_temp, [[1]], axis=0)
        layer3 = np.dot(np.transpose(self.Theta2), layer2)
        cost = self.Sigmoid(layer3[0, 0])   # sigmoid cost in [0,1]
        return (layer1, layer2, cost)

    def TestStep(self):
        outL = self.ForwardPropagate('L')
        outR = self.ForwardPropagate('R')
        if outL[-1] < outR[-1]:
            self.game.ChangeDirection('L')
        else:
            self.game.ChangeDirection('R')
        return self.game.GameOver()


# =========================
# Training bot (Cost TD learning)
# =========================

class BotTrain(Bot):
    """
    Train the same way original report describes (value/cost learning),
    so that your learned agent matches the old high-score behavior.

    Update target:
      if terminal: estimate = GameOverCost
      else:
         estimate = min(cost(s',L), cost(s',R)) ** discount
         if Kill: estimate *= gamma   (encourage surviving & scoring)

    expected = (1-a)*cost(s,a) + a*estimate
    SGD by backprop on (cost - expected)
    """

    def __init__(
        self,
        GameParameters,
        HiddenSize=12,
        gamma=0.8,
        GameOverCost=1.0,
        NSim=400,
        NTest=400,
        TestTreshold=None,
        NumberOfSessions=100,
        Inertia=0.8,
        p=0.0,
        a=1.0,
        epsilon=0.05,
        epsilon_decay_rate=0.0,
        discount=0.999,
        p_decay_rate=0.5,
        sgd_lr=0.01,
        verbose=False,
    ):
        Theta1 = np.random.uniform(-1.0, 1.0, (2 * GameParameters["N"] + 1, HiddenSize))
        Theta2 = np.random.uniform(-1.0, 1.0, (HiddenSize + 1, 1))
        game = Game(**GameParameters)
        super(BotTrain, self).__init__(Theta1, Theta2, game)

        self.GameParameters = dict(GameParameters)
        self.HiddenSize = HiddenSize

        self.gamma = gamma
        self.GameOverCost = float(GameOverCost)
        self.NSim = int(NSim)
        self.NTest = int(NTest)
        self.TestTreshold = TestTreshold
        self.NumberOfSessions = NumberOfSessions

        self.Inertia = float(Inertia)
        self.p = float(p)
        self.a = float(a)

        self.epsilon = float(epsilon)
        self.epsilon_decay_rate = float(epsilon_decay_rate)
        self.discount = float(discount)
        self.p_decay_rate = float(p_decay_rate)

        self.sgd_lr = float(sgd_lr)
        self.verbose = verbose

        self.counter = []
        self.best_score = 0

    # ---------- save ----------
    def _ensure_data_dir(self):
        if not os.path.isdir("Data"):
            os.makedirs("Data")

    def _learn_params_dict(self):
        return {
            "HiddenSize": self.HiddenSize,
            "gamma": self.gamma,
            "GameOverCost": self.GameOverCost,
            "NSim": self.NSim,
            "NTest": self.NTest,
            "TestTreshold": self.TestTreshold,
            "NumberOfSessions": self.NumberOfSessions,
            "Inertia": self.Inertia,
            "p": self.p,
            "a": self.a,
            "epsilon": self.epsilon,
            "epsilon_decay_rate": self.epsilon_decay_rate,
            "discount": self.discount,
            "p_decay_rate": self.p_decay_rate,
            "sgd_lr": self.sgd_lr,
            "mode": "LEGACY_COST_TD"
        }

    def save_npz_with_json(self, path):
        self._ensure_data_dir()
        gp_json = json.dumps(self.GameParameters, ensure_ascii=False)
        lp_json = json.dumps(self._learn_params_dict(), ensure_ascii=False)
        np.savez(
            path,
            GameParameters=np.array(self.GameParameters, dtype=object),
            GameParameters_json=np.array(gp_json),
            LearnParameters_json=np.array(lp_json),
            Theta1=self.Theta1,
            Theta2=self.Theta2,
        )

    # ---------- backprop ----------
    def BackPropagate(self, output, expected, layer1, layer2):
        # output is cost in [0,1]
        delta3 = output - expected
        delta2 = delta3 * self.Theta2
        for i in range(self.HiddenSize):
            delta2[i, 0] *= layer2[i, 0] * (1 - layer2[i, 0])

        for i in range(2 * self.game.N + 1):
            for j in range(self.HiddenSize):
                self.Theta1[i, j] -= self.sgd_lr * layer1[i, 0] * delta2[j, 0]
        for i in range(self.HiddenSize + 1):
            self.Theta2[i, 0] -= self.sgd_lr * delta3 * layer2[i, 0]

    # ---------- learning step ----------
    def ReinforcedLearningStep(self):
        # mix exploration / inertia / learned
        t = random.random()
        if t < 1 - self.p:
            tt = random.random()
            if tt < self.Inertia:
                output = self.ForwardPropagate(self.game.Direction)
            else:
                new_direction = random.choice(['L', 'R'])
                output = self.ForwardPropagate(new_direction)
                self.game.ChangeDirection(new_direction)
        else:
            outL = self.ForwardPropagate('L')
            outR = self.ForwardPropagate('R')
            if outL[-1] < outR[-1]:
                output = outL
                self.game.ChangeDirection('L')
            else:
                output = outR
                self.game.ChangeDirection('R')

        (Update, Kill, Over) = self.game.UpdateStep()

        if Over:
            estimate = self.GameOverCost
        else:
            estL = self.ForwardPropagate('L')
            estR = self.ForwardPropagate('R')
            estimate = min(estL[-1], estR[-1]) ** self.discount
            if Kill:
                estimate *= self.gamma

        expected = (1 - self.a) * output[-1] + self.a * estimate
        self.BackPropagate(output[-1], expected, output[0], output[1])
        return (Update, Kill, Over)

    def Training(self):
        train_scores = []
        for _ in range(self.NSim):
            stop = False
            while not stop:
                (_, _, stop) = self.ReinforcedLearningStep()
            train_scores.append(self.game.counter)
            self.game = Game(**self.GameParameters)
        return train_scores

    def Testing(self):
        scores = []
        for _ in range(self.NTest):
            stop = False
            while not stop:
                stop = self.TestStep()
                self.game.UpdateStep()
            scores.append(self.game.counter)
            self.game = Game(**self.GameParameters)

        m1 = sum(scores) / (len(scores) + 0.0)
        m2 = float(np.median(scores))
        self.counter.append((m1, m2))

        if m1 > self.best_score:
            self.best_score = m1
            N = int(self.GameParameters.get("N", 0))
            np.savez(
                os.path.join("Data", "parameters_best%d.npz" % N),
                GameParameters=self.GameParameters,
                Theta1=self.Theta1,
                Theta2=self.Theta2,
            )

    def TrainSession(self):
        self.Testing()
        keep_going = True
        i = 0
        while keep_going:
            i += 1
            print("\nN:", self.game.N)
            print("Session:", i)

            train_scores = self.Training()
            print("Train mean/median:",
                  sum(train_scores) / (len(train_scores) + 0.0),
                  float(np.median(train_scores)))

            self.Testing()
            print("Test Results:", self.counter)

            new, old = self.counter[-1][-1], self.counter[-2][-1]
            if new > 0:
                self.sgd_lr *= (old / new) ** self.epsilon_decay_rate

            self.p = 1 - (1 - self.p) * ((old / new) ** self.p_decay_rate) if new > 0 else self.p
            if self.p < 0:
                self.p = 0.0
            print("p:", self.p, "sgd_lr:", self.sgd_lr)

            if self.TestTreshold is None and self.NumberOfSessions is not None:
                if i >= self.NumberOfSessions:
                    keep_going = False
            elif self.TestTreshold is not None:
                if self.counter[-1][-1] >= self.TestTreshold:
                    keep_going = False
