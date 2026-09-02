"""
Training module — Continuous self-RL training loop, walk-forward evaluation,
and multi-agent tournament for model promotion.

Components:
  - continuous_trainer: Background daemon that trains while inference runs
  - walk_forward_evaluator: Statistical promotion gate (must beat baselines)
  - crucible_tournament: 3-5 agent focused training competition
"""
