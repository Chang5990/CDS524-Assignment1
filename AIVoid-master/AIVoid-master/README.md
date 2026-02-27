1. AIVoid
The AI agent uses Q-learning (TD bootstrap) with a two-layer NumPy neural network to approximate action values.
During inference (rungame1.py), each frame it computes scores for Left and Right and selects the better action. During training (training.py + Bots.py), the network parameters are updated toward a TD target derived from the next state.
2. How to Run (Demo)：
    (1) Run: rungame1.py
    (2) In the popup window, choose: N (number of obstacles): 4 / 5 / 6
                                     Speed: fast / medium / slow
                                     Mode:  Human: control with Left/Right, press SPACE to start/restart
                                            AI best: load the best model saved during training (parameters_best*.npz)
                                            AI post train: load the final model (parametersN.npz)
                                            AI pre train: random weights baseline (for comparison)

    (3) The right-side HUD shows required information: state / action / reward / penalty.

3. Dependencies:
   (1) Python 3 (recommended)
   (2) NumPy
   (3) Tkinter

