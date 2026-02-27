# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import json
import random
import numpy as np

# Tkinter: Python3/2 compatible
try:
    from tkinter import *
    try:
        from tkinter import messagebox
    except Exception:
        messagebox = None
except ImportError:
    from tkinter import *
    try:
        import tkMessageBox as messagebox
    except Exception:
        messagebox = None

from Game import Game
# -------------------------
# Utils
# -------------------------
def show_error(msg):
    if messagebox is None:
        print("[ERROR]", msg)
        return
    root = Tk()
    root.withdraw()
    try:
        messagebox.showerror("AIVoid", msg)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def stable_sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def build_layer1(game, action):
    """
    State vector:
      [x1/HW, y1/H, x2/HW, y2/H, ..., bias=1]
    Symmetry:
      if action == 'L' flip x coords (0,2,4,...)
    """
    state = []
    for aster in game.asteroids:
        state.append(aster[0] / (game.Halfwidth + 0.0))
        state.append(aster[1] / (game.Height + 0.0))
    state.append(1.0)

    if action == 'L':
        for i in range(game.N):
            state[2 * i] *= -1.0

    return np.array(state, dtype=np.float64).reshape((-1, 1))

def forward_linear_q(Theta1, Theta2, game, action):
    """NEW style: output is linear Q value (real number), greedy uses max."""
    layer1 = build_layer1(game, action)
    h = stable_sigmoid(np.dot(Theta1.T, layer1))
    layer2 = np.vstack([h, np.array([[1.0]], dtype=np.float64)])
    out = np.dot(Theta2.T, layer2)  # (1,1)
    return float(out[0, 0])


def forward_sigmoid_cost(Theta1, Theta2, game, action):
    """LEGACY style: output is sigmoid(cost) in [0,1], greedy uses min."""
    layer1 = build_layer1(game, action)
    h = stable_sigmoid(np.dot(Theta1.T, layer1))
    layer2 = np.vstack([h, np.array([[1.0]], dtype=np.float64)])
    out = np.dot(Theta2.T, layer2)
    return float(stable_sigmoid(out)[0, 0])

class PolicyAdapter(object):
    """
    Four modes to interpret Theta:
      - legacy_min_cost: sigmoid(cost), pick MIN  (best for your best6 model)
      - new_max_q: linear(q), pick MAX
      - sigmoid_max: sigmoid(cost), pick MAX (sanity)
      - linear_min: linear(q), pick MIN (sanity)
    """

    def __init__(self, Theta1, Theta2, mode):
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.mode = mode

    def choose_action(self, game):
        if self.mode in ("legacy_min_cost", "sigmoid_max"):
            vL = forward_sigmoid_cost(self.Theta1, self.Theta2, game, 'L')
            vR = forward_sigmoid_cost(self.Theta1, self.Theta2, game, 'R')
            if self.mode == "legacy_min_cost":
                return 'L' if vL < vR else 'R'
            else:
                return 'L' if vL > vR else 'R'
        else:
            vL = forward_linear_q(self.Theta1, self.Theta2, game, 'L')
            vR = forward_linear_q(self.Theta1, self.Theta2, game, 'R')
            if self.mode == "new_max_q":
                return 'L' if vL > vR else 'R'
            else:
                return 'L' if vL < vR else 'R'

    def step(self, game):
        a = self.choose_action(game)
        game.ChangeDirection(a)


def default_game_parameters(N, SleepTime=10):
    return {
        'N': int(N),
        'DownSideRatio': 3,
        'SleepTime': int(SleepTime),
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


def load_model_npz(path):
    """
    Robust loader:
      - Prefer GameParameters_json if exists
      - Else try pickle GameParameters
      - If pickle fails (cross-numpy), fall back to defaults inferred from Theta shape
    """
    arrays = np.load(path, allow_pickle=True)
    Theta1 = arrays['Theta1']
    Theta2 = arrays['Theta2']

    gp = None
    if 'GameParameters_json' in arrays.files:
        s = arrays['GameParameters_json']
        try:
            s = s.item()
        except Exception:
            pass
        gp = json.loads(str(s))
    elif 'GameParameters' in arrays.files:
        try:
            gp = arrays['GameParameters'].item()
        except Exception:
            gp = None

    if gp is None:
        in_dim = int(Theta1.shape[0])
        N = max(1, (in_dim - 1) // 2)
        gp = default_game_parameters(N=N)

    return gp, Theta1, Theta2


def quick_benchmark_policy(Theta1, Theta2, GameParameters, episodes=10, maxscore_cap=200):
    """
    Fast policy selection:
      run small number of episodes with MaxScore cap
    """
    modes = ["legacy_min_cost", "new_max_q", "sigmoid_max", "linear_min"]
    scores = {}

    bench_params = dict(GameParameters)
    bench_params['MaxScore'] = maxscore_cap

    for mode in modes:
        policy = PolicyAdapter(Theta1, Theta2, mode)
        alist = []
        for i in range(episodes):
            random.seed(1234 + i)
            np.random.seed(1234 + i)

            g = Game(**bench_params)
            stop = False
            steps = 0
            while not stop and steps < 20000:
                policy.step(g)
                (_, _, stop) = g.UpdateStep()
                steps += 1
            alist.append(int(g.counter))
        scores[mode] = float(sum(alist)) / (len(alist) + 0.0)

    best_mode = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_mode, scores


# -------------------------
# UI: Choose Mode
# -------------------------
class Choose:
    def __init__(self):
        self.master = Tk()
        self.master.title("AIVoid - Choose Mode")

        howmany_options = [6, 5, 4]
        howfast_options = ['fast', 'medium', 'slow']

        self.howmany = IntVar(value=howmany_options[0])
        self.howfast = StringVar(value=howfast_options[0])
        self.who = None

        Label(self.master, font=("Purisa", 13), text="Number of objects (N)").pack(padx=10, pady=(10, 0))
        OptionMenu(self.master, self.howmany, *howmany_options).pack(padx=10, pady=5)

        Label(self.master, font=("Purisa", 13), text="Game speed").pack(padx=10, pady=(10, 0))
        OptionMenu(self.master, self.howfast, *howfast_options).pack(padx=10, pady=5)

        Label(self.master, font=("Purisa", 13), text="Player").pack(padx=10, pady=(10, 0))

        btn = Frame(self.master)
        btn.pack(padx=10, pady=10)

        Button(btn, font=("Purisa", 12), text="Human", command=self.human).pack(side=LEFT, padx=3)
        Button(btn, font=("Purisa", 12), text="AI post train", command=self.ai).pack(side=LEFT, padx=3)
        Button(btn, font=("Purisa", 12), text="AI best", command=self.ai_best).pack(side=LEFT, padx=3)
        Button(btn, font=("Purisa", 12), text="AI pre train", command=self.dumb_ai).pack(side=LEFT, padx=3)

        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.who = None
        self.master.destroy()

    def human(self):
        self.who = 'human'
        self.master.destroy()

    def dumb_ai(self):
        self.who = 'dumb_ai'
        self.master.destroy()

    def ai(self):
        self.who = 'ai'
        self.master.destroy()

    def ai_best(self):
        self.who = 'ai_best'
        self.master.destroy()


# -------------------------
# Gameplay UI with rubric-required HUD (3.2)
# -------------------------
class PlayBase:
    """
    Rubric 3.2 requirement:
      UI displays current state, agent actions, and rewards/penalties.
    We implement a RIGHT-side panel with:
      - state: nearest K asteroids
      - action: L/R
      - reward: +1 when score increases
      - penalty: -1 when collision (terminal)
      - parameters: N, speed, noise, model/policy
    """

    def __init__(self, GameParameters, title="AIVoid", model_path=None, policy_mode=None, k_state=3):
        self.GameParameters = dict(GameParameters)
        self.game = Game(**self.GameParameters)

        self.model_path = model_path
        self.policy_mode = policy_mode
        self.k_state = int(k_state)

        self.x = self.game.GlobalWidth / 2
        self.y = self.game.GlobalHeight - self.game.Thickness

        self.master = Tk()
        self.master.title(title)

        # Layout: canvas left, info panel right
        self.root_frame = Frame(self.master)
        self.root_frame.pack(fill=BOTH, expand=True)

        self.canvas = Canvas(self.root_frame, bg="black",
                             width=self.game.GlobalWidth, height=self.game.GlobalHeight)
        self.canvas.pack(side=LEFT)

        self.panel = Frame(self.root_frame, width=320)
        self.panel.pack(side=RIGHT, fill=Y)

        self.info_title = Label(self.panel, text="HUD (State / Action / Reward)", font=("Purisa", 12, "bold"))
        self.info_title.pack(anchor="w", padx=10, pady=(10, 5))

        self.info_text = StringVar()
        self.info_label = Label(self.panel, textvariable=self.info_text, justify=LEFT,
                                font=("Consolas", 10), wraplength=300)
        self.info_label.pack(anchor="w", padx=10, pady=5)

        self.help = Label(self.panel, text="SPACE: start/restart\nHuman: Left/Right",
                          font=("Purisa", 10))
        self.help.pack(anchor="w", padx=10, pady=(10, 10))

        # loop state
        self.running = False
        self.over = False
        self.after_id = None
        self.step_count = 0

        # reward/penalty bookkeeping
        self.last_reward = 0
        self.last_penalty = 0

        # draw world
        self._draw_world()
        self._refresh_info()

        # bindings
        self.canvas.focus_set()
        self.canvas.bind("<KeyPress-space>", self._space_press)

    def _nearest_asteroids_lines(self):
        if not self.game.asteroids:
            return ["State: (no asteroids)"]

        # nearest to ship: smaller y means closer to ship (in your coordinate system)
        ast = sorted(self.game.asteroids, key=lambda a: a[1])
        lines = ["State (nearest obstacles):"]
        for i, a in enumerate(ast[:self.k_state], 1):
            dx, dy = float(a[0]), float(a[1])
            xn = dx / (self.game.Halfwidth + 0.0)
            yn = dy / (self.game.Height + 0.0)
            lines.append("  %d) dx=%7.2f  dy=%7.2f  | x'=% .3f y'=% .3f" % (i, dx, dy, xn, yn))
        return lines

    def _refresh_info(self):
        gp = self.GameParameters
        lines = []
        lines.append("Action(Direction): %s" % self.game.Direction)
        lines.append("Score: %d   Step: %d" % (int(self.game.counter), int(self.step_count)))
        lines.append("Reward(+1 destroy): %s" % str(self.last_reward))
        lines.append("Penalty(-1 collision): %s" % str(self.last_penalty))
        lines.append("")
        lines.append("Parameters:")
        lines.append("  N=%s  Speed(SleepTime)=%s" % (gp.get("N"), gp.get("SleepTime")))
        lines.append("  DownSideRatio=%s" % gp.get("DownSideRatio"))
        lines.append("  Noise(RandomTreshold)=%s" % gp.get("RandomTreshold"))
        lines.append("")
        if self.model_path:
            lines.append("Model file:")
            lines.append("  %s" % self.model_path)
        if self.policy_mode:
            lines.append("Policy mode: %s" % self.policy_mode)
        lines.append("")
        lines.extend(self._nearest_asteroids_lines())

        self.info_text.set("\n".join(lines))

    def _space_press(self, event=None):
        if self.over:
            self.restart_game()
            self.start_loop()
            return
        if not self.running:
            self.start_loop()

    def start_loop(self):
        self.running = True
        self._tick()

    def stop_loop(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.master.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def tick_pre_step(self):
        """Override in subclasses (AI chooses action before step)."""
        pass

    def _tick(self):
        if not self.running:
            return
        if self.over:
            self.stop_loop()
            return

        self.tick_pre_step()
        over_now = self.run_step()
        if over_now:
            self.stop_loop()
            return

        self.after_id = self.master.after(self.game.SleepTime, self._tick)

    def restart_game(self):
        self.stop_loop()
        self.canvas.delete("all")
        self.game = Game(**self.GameParameters)
        self.x = self.game.GlobalWidth / 2
        self.y = self.game.GlobalHeight - self.game.Thickness
        self.over = False
        self.step_count = 0
        self.last_reward = 0
        self.last_penalty = 0
        self._draw_world()
        self._refresh_info()
        self.on_restart()

    def on_restart(self):
        pass

    def _draw_world(self):
        # ship + borders (3 copies)
        for i in range(3):
            cx, cy = self.x + (i - 1) * self.game.GlobalWidth, self.y
            self.canvas.create_oval(cx - self.game.R, cy + self.game.R,
                                    cx + self.game.R, cy - self.game.R,
                                    fill="blue", width=0, tag='S')
            self.canvas.create_rectangle(cx - self.game.Halfwidth - self.game.Thickness - self.game.r, cy,
                                         cx + self.game.Halfwidth + self.game.Thickness + self.game.r, cy + self.game.Thickness,
                                         fill="white", width=0, tag='S')
            self.canvas.create_rectangle(cx - self.game.Halfwidth - self.game.Thickness - self.game.r, cy - self.game.Height,
                                         cx - self.game.Halfwidth - self.game.r, cy + self.game.Thickness,
                                         fill="white", width=0, tag='S')
            self.canvas.create_rectangle(cx + self.game.Halfwidth + self.game.r, cy - self.game.Height,
                                         cx + self.game.Halfwidth + self.game.Thickness + self.game.r, cy + self.game.Thickness,
                                         fill="white", width=0, tag='S')

        # asteroids
        for aster in self.game.asteroids:
            for i in range(3):
                cx, cy = self.x + (i - 1) * self.game.GlobalWidth + aster[0], self.y - aster[1]
                self.canvas.create_oval(cx - self.game.r, cy + self.game.r,
                                        cx + self.game.r, cy - self.game.r,
                                        fill="red", width=0, tag='A')

    def run_step(self):
        # reward bookkeeping
        prev_score = int(self.game.counter)
        self.last_reward = 0
        self.last_penalty = 0
        self.step_count += 1

        # ship movement (visual only)
        if self.game.Direction == 'L':
            self.canvas.move('S', -1, 0)
            self.x -= 1
            if self.x < (-1) * self.game.GlobalWidth / 2:
                self.x += self.game.GlobalWidth
                self.canvas.move('S', self.game.GlobalWidth, 0)

        if self.game.Direction == 'R':
            self.canvas.move('S', 1, 0)
            self.x += 1
            if self.x > 3 * self.game.GlobalWidth / 2:
                self.x -= self.game.GlobalWidth
                self.canvas.move('S', (-1) * self.game.GlobalWidth, 0)

        # move asteroids visuals down
        self.canvas.move('A', 0, self.game.DownSideRatio)

        (Update, Kill, Over) = self.game.UpdateStep()

        # reward: +1 when score increases
        new_score = int(self.game.counter)
        if new_score > prev_score:
            self.last_reward = new_score - prev_score

        # refresh asteroids visuals if needed
        if Update:
            self.canvas.delete('A')
            for aster in self.game.asteroids:
                for i in range(3):
                    cx, cy = self.x + (i - 1) * self.game.GlobalWidth + aster[0], self.y - aster[1]
                    self.canvas.create_oval(cx - self.game.r, cy + self.game.r,
                                            cx + self.game.r, cy - self.game.r,
                                            fill="red", width=0, tag='A')

        # penalty on collision terminal
        if Over and not self.over:
            self.over = True
            # MaxScore terminal is not a penalty; collision is a penalty
            if self.game.MaxScore is None or self.game.counter < self.game.MaxScore:
                self.last_penalty = -1

            # show overlay text
            self.canvas.create_text(
                self.game.GlobalWidth / 2,
                self.game.GlobalHeight / 2,
                fill="white", font=("Purisa", 24, "bold"),
                text="GAME OVER\nPress SPACE to restart"
            )

        self._refresh_info()
        return Over


class PlayHuman(PlayBase):
    def __init__(self, GameParameters):
        super(PlayHuman, self).__init__(GameParameters, title="AIVoid - Human")
        self.canvas.bind("<Left>", self.left)
        self.canvas.bind("<Right>", self.right)

    def on_restart(self):
        self.canvas.bind("<Left>", self.left)
        self.canvas.bind("<Right>", self.right)

    def left(self, event=None):
        self.game.ChangeDirection('L')
        self._refresh_info()

    def right(self, event=None):
        self.game.ChangeDirection('R')
        self._refresh_info()


class PlayAI(PlayBase):
    def __init__(self, Theta1, Theta2, GameParameters, model_path, policy_mode, title="AIVoid - AI"):
        super(PlayAI, self).__init__(GameParameters, title=title, model_path=model_path, policy_mode=policy_mode)
        self.policy = PolicyAdapter(Theta1, Theta2, policy_mode)

    def tick_pre_step(self):
        self.policy.step(self.game)


def pick_model_path(who, N):
    N = int(N)
    if who == "ai_best":
        p = os.path.join("Data", "parameters_best%d.npz" % N)
        if os.path.exists(p):
            return p
        p2 = os.path.join("Data", "parameters_best.npz")
        if os.path.exists(p2):
            return p2
        return os.path.join("Data", "parameters%d.npz" % N)

    if who in ("ai", "dumb_ai"):
        return os.path.join("Data", "parameters%d.npz" % N)

    return None


# -------------------------
# MAIN
# -------------------------
choose = Choose()
choose.master.mainloop()

who = choose.who
if who is None:
    print("No player mode selected.")
    raise SystemExit(0)

howmany = int(choose.howmany.get())
SpeedDict = {'fast': 5, 'medium': 10, 'slow': 15}
howfast = int(SpeedDict.get(choose.howfast.get(), 10))

if who == "human":
    gp = default_game_parameters(N=howmany, SleepTime=howfast)
    app = PlayHuman(gp)
    app.master.mainloop()
    print("Human Score:", app.game.counter)
    raise SystemExit(0)

model_path = pick_model_path(who, howmany)
if model_path is None or (not os.path.exists(model_path)):
    show_error("Model file not found:\n%s" % str(model_path))
    raise SystemExit(1)

# load model
gp, Theta1, Theta2 = load_model_npz(model_path)

# ensure N matches Theta1
thetaN = max(1, (int(Theta1.shape[0]) - 1) // 2)
if int(gp.get("N", thetaN)) != thetaN:
    gp["N"] = thetaN

# speed from UI
gp["SleepTime"] = howfast

# dumb AI: randomize weights but keep policy auto-selection
if who == "dumb_ai":
    Theta1 = np.random.uniform(-1.0, 1.0, Theta1.shape)
    Theta2 = np.random.uniform(-1.0, 1.0, Theta2.shape)
    title = "AIVoid - AI pre train"
else:
    title = "AIVoid - AI best" if who == "ai_best" else "AIVoid - AI post train"

best_mode, score_map = quick_benchmark_policy(Theta1, Theta2, gp, episodes=10, maxscore_cap=200)
print("Loaded model:", model_path)
print("Auto policy scores:", score_map)
print("Selected policy:", best_mode)
app = PlayAI(Theta1, Theta2, gp, model_path=model_path, policy_mode=best_mode, title=title)
app.master.mainloop()
print(title.replace("AIVoid - ", "") + " Score:", app.game.counter)
