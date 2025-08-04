✅ Milestone 3: Add Baseline Classical/Manual Policy — Checklist

## A. Implement Manual Baseline Policies

Create src/agents/manual_policies.py (if not already made)

Implement always_max_policy (push T/F to max)

Implement random_policy (sample random actions)

Implement proportional_policy (adjust T/F based on error to target)

(Optional) Implement a step-ramp policy (T/F step up or down after fixed intervals)

## B. Policy Evaluation Utility

Create or expand a utility function evaluate_policy(policy_fn, env, n_episodes=10)

Resets env, runs policy until done, logs final thickness error

Collects and returns error/statistics (and optionally trajectories)

Save in src/agents/compare_policies.py or as a new notebook notebooks/compare_policies.ipynb

## C. RL Baseline for Comparison

Train RL agent (e.g., DQN with train_dqn) for fair comparison

Wrap model with rl_policy(env, state) for evaluation

## D. Run Policy Benchmarks

For each policy (always max, random, proportional, RL):

Run N episodes (e.g., N=10)

Collect final errors for each run

(Optional) Save/control trajectories of T, F, h

## E. Visualization and Analysis

Plot results (boxplot or histogram) for final error across all policies

(Optional) Plot sample T/F/h trajectories per policy

(Optional) Quantify “control effort” (e.g., total |ΔT| + |ΔF| per episode)

## F. Documentation & Interpretation

Markdown summary/table of results:

Mean and std of errors for each policy

Notes on qualitative behavior (overshoot, oscillation, stability)

Can RL outperform or adapt better than classical policies?

Explain differences for a skeptical process engineer

## G. (Optional) Stress Test

Repeat comparisons for multiple kinetic regimes (different k₀/Eₐ) to see if RL advantage holds

Comment/document any edge cases or surprising results
