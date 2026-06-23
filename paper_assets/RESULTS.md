# Benchmark results

Physics-informed dynamics models for underactuated swing-up. Five models (Vanilla MLP, Lagrangian NN, Hamiltonian NN, Vanilla GP, Lagrangian GP) on Acrobot and Pendubot, across four axes.


## Dynamics — Acrobot (largest budget n_train=4000)

| Model | One-step MSE ↓ | Valid steps ↑ | Energy drift ↓ | Diverged |
|---|---|---|---|---|
| Hamiltonian_NN | 1.61e-02 | 10.7 | 9.63e+02 | 20% |
| Lagrangian_GP | 1.45e-03 | 52.9 | 5.66e+00 | 0% |
| Lagrangian_NN | 1.39e-03 | 27.3 | 7.73e+00 | 0% |
| Polynomial | 4.21e-02 | 13.2 | 6.72e+02 | 36% |
| Structured_HNN | 1.39e-03 | 27.3 | 7.73e+00 | 0% |
| Vanilla_GP | 8.46e-03 | 67.6 | 9.70e+00 | 0% |
| Vanilla_MLP | 6.13e-03 | 25.8 | 1.98e+01 | 0% |

## Dynamics — Pendubot (largest budget n_train=4000)

| Model | One-step MSE ↓ | Valid steps ↑ | Energy drift ↓ | Diverged |
|---|---|---|---|---|
| Hamiltonian_NN | 1.55e-02 | 12.2 | 9.95e+02 | 32% |
| Lagrangian_GP | 2.27e-03 | 58.6 | 6.72e+00 | 0% |
| Lagrangian_NN | 2.57e-03 | 26.9 | 9.95e+00 | 0% |
| Polynomial | 4.14e-02 | 11.9 | 2.98e+02 | 16% |
| Structured_HNN | 2.57e-03 | 26.9 | 9.95e+00 | 0% |
| Vanilla_GP | 8.83e-03 | 71.4 | 1.13e+01 | 0% |
| Vanilla_MLP | 7.16e-03 | 28.0 | 1.87e+01 | 0% |

## Swing-up control (Axis 4) — Acrobot

| Model | Mean tip height (0–1) | Success rate |
|---|---|---|
| Oracle | 0.965 | 100% |
| Hamiltonian_NN | 0.506 | 0% |
| Lagrangian_GP | 0.628 | 25% |
| Lagrangian_NN | 0.628 | 25% |
| Structured_HNN | 0.629 | 25% |

## Swing-up control (Axis 4) — Pendubot

| Model | Mean tip height (0–1) | Success rate |
|---|---|---|
| Oracle | 0.956 | 100% |
| Hamiltonian_NN | 0.659 | 10% |
| Lagrangian_GP | 0.647 | 15% |
| Lagrangian_NN | 0.647 | 15% |
| Structured_HNN | 0.643 | 15% |

## Figures

![sweep_Acrobot.png](figures/sweep_Acrobot.png)

![sweep_Pendubot.png](figures/sweep_Pendubot.png)

![control_Acrobot.png](figures/control_Acrobot.png)

![control_Pendubot.png](figures/control_Pendubot.png)
