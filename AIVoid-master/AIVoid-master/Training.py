# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import shutil
from Bots import BotTrain


def ensure_data_dir():
    if not os.path.isdir("Data"):
        os.makedirs("Data")


def train_one_N(N, game_params_base, learn_params_base):
    GameParameters = dict(game_params_base)
    GameParameters["N"] = int(N)

    LearnParameters = dict(learn_params_base)

    out_path = os.path.join("Data", "parameters%d.npz" % N)
    best_path = os.path.join("Data", "parameters_best%d.npz" % N)

    print("\n==============================")
    print("=== Training LEGACY cost-TD for N =", N, "===")
    print("Output:", out_path)
    print("GameParameters:", GameParameters)
    print("LearnParameters:", LearnParameters)

    bot = BotTrain(GameParameters=GameParameters, **LearnParameters)
    bot.TrainSession()

    bot.save_npz_with_json(out_path)
    print("Saved FINAL model:", out_path)

    if not os.path.exists(best_path):
        shutil.copyfile(out_path, best_path)
        print("Best missing, copied FINAL -> BEST:", best_path)

    return out_path, best_path


def main():
    ensure_data_dir()

    game_params_base = {
        'DownSideRatio': 3,
        'SleepTime': 5,
        'R': 25,
        'r': 5,
        'Height': 400,
        'Halfwidth': 200,
        'GlobalHeight': 600,
        'GlobalWidth': 800,
        'Thickness': 20,
        'RandomTreshold': 0.2,
        'RandomStep': 1,
        'RandomVertTreshold': 0.2,
        'RandomVertStep': 1,
        'MaxScore': None
    }

    # 这套参数更接近原项目
    learn_params_base = {
        "HiddenSize": 12,
        "gamma": 0.8,
        "GameOverCost": 3.8,        #1.0
        "NSim": 400,
        "NTest": 400,
        "TestTreshold": None,
        "NumberOfSessions": 100,
        "Inertia": 0.8,
        "p": 0.0,
        "a": 1.0,
        "epsilon": 0.05,
        "epsilon_decay_rate": 0.0,
        "discount": 0.999,
        "p_decay_rate": 0.5,
        "sgd_lr": 0.01,
        "verbose": False
    }

    N_list = [4, 5, 6]
    best_map = {}
    for N in N_list:
        _, best_path = train_one_N(N, game_params_base, learn_params_base)
        best_map[N] = best_path

    # alias: parameters_best.npz = best5
    alias_N = 5
    alias_src = best_map.get(alias_N)
    alias_dst = os.path.join("Data", "parameters_best.npz")
    if alias_src and os.path.exists(alias_src):
        shutil.copyfile(alias_src, alias_dst)
        print("\n[Alias] Copied", alias_src, "->", alias_dst)

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
