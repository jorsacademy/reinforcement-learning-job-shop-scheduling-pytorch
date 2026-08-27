# Reinforcement Learning for Job-Shop Scheduling with PyTorch PPO

A from-scratch reinforcement-learning project for the classical **Job-Shop Scheduling Problem (JSSP)**.

The project keeps the Industrial Engineering objective explicit:

```text
jobs + operation routes + processing times
                 ↓
serial schedule-generation environment
                 ↓
masked Transformer actor-critic
                 ↓
PPO
                 ↓
feasible machine schedule
                 ↓
makespan
```

The neural policy is evaluated against classical dispatching rules and an independent OR-Tools CP-SAT model.

## Problem

Each job has an ordered sequence of operations. Every operation:

- must run on one specified machine;
- has an integer processing time;
- cannot start before the previous operation of the same job finishes;
- cannot overlap another operation on the same machine.

The objective is:

```text
minimize makespan
```

or equivalently the completion time of the last finished job.

The synthetic generator uses permutation job shops: every job visits every machine exactly once, but route order and processing times differ by job.

## Environment semantics

`JobShopEnv` is a Gymnasium-compatible serial schedule-generation environment.

An action chooses **which unfinished job contributes its next operation to the priority/dispatch sequence**.

That operation is appended to its required machine and starts at:

```text
max(job_ready_time, machine_ready_time)
```

This always produces a precedence-feasible and machine-feasible semi-active schedule.

The action mask is therefore:

```text
1 = job still has an unscheduled operation
0 = job is complete
```

The policy never receives an illegal completed-job action during normal sampling because invalid logits are masked to `-inf`.

This is a priority-sequence scheduling environment. It should not be confused with a real-time shop-floor event simulator in which only operations executable at the current wall-clock instant are actions.

## Exact objective-aligned reward

Let `C_t` be the partial schedule makespan after decision `t`.

The environment reward is:

```text
r_t = C_(t-1) - C_t
```

Therefore:

```text
sum_t r_t = - final_makespan
```

exactly.

PPO is deliberately restricted to:

```text
gamma = 1.0
```

so discounted return does not silently change the scheduling objective.

For numerical conditioning, the PPO implementation may multiply all rewards by one fixed positive constant before the critic/advantage calculation. A positive constant scaling does not change the ordering of policies for a fixed problem family.

A regression test checks the raw telescoping identity directly.

## Observation

Every job is represented with dynamic and remaining-route information.

Core dynamic features:

- next operation processing time;
- job ready time;
- required machine ready time;
- earliest possible start of the next operation;
- total remaining work;
- number of remaining operations;
- completion flag.

Routing features:

- one-hot next required machine;
- remaining processing time on every machine;
- relative future position of every remaining machine in the job route.

Global features:

- current partial makespan;
- fraction of operations already scheduled;
- mean machine ready time;
- maximum machine ready time.

The policy sees scheduling state and problem data, but no exact-solver label.

## Actor-Critic

The policy is implemented directly in PyTorch.

```text
per-job features
      ↓
shared embedding
      ↓
Transformer encoder across jobs
      ↓
job embeddings + pooled schedule context
      ↓
masked actor logits

pooled job context + global state
      ↓
critic value
```

There is no Stable-Baselines3 dependency. PPO rollout collection, GAE, clipping, entropy regularization, value loss, gradient clipping and validation checkpoint selection are implemented in this repository.

## PPO

The clipped surrogate is the standard PPO form:

```text
ratio = exp(log π_new(a|s) - log π_old(a|s))

L_policy =
-mean(
    min(
        ratio * advantage,
        clip(ratio, 1-eps, 1+eps) * advantage
    )
)
```

Training instances are generated **on the fly** every epoch.

A disjoint fixed validation set is used to select the best greedy checkpoint by validation makespan. The held-out test instances are generated from another disjoint seed range.

This avoids selecting the final network on the test benchmark.

## Classical dispatching baselines

The repository includes four non-learning policies:

```text
SPT             shortest next processing time
MWKR            most work remaining
Earliest-start  smallest current earliest start time
Random          seeded random feasible job
```

Every baseline is passed through the same schedule generator and the resulting schedules are feasibility-audited.

## Exact OR-Tools CP-SAT oracle

For small held-out instances, an independent CP-SAT model is available.

Each operation receives:

```text
start variable
end variable
interval variable
```

Constraints:

```text
job precedence
machine AddNoOverlap
```

Objective:

```text
minimize max(final operation end time)
```

Only instances for which CP-SAT reports `OPTIMAL` are used to report an exact optimality gap.

If the solver reports only `FEASIBLE` within a time limit, that result is not relabeled as an exact optimum.

## Independent tiny exhaustive oracle

The regression suite does not trust CP-SAT alone.

For tiny instances with at most ten operations, every precedence-feasible job-dispatch sequence is enumerated. Each sequence is transformed into a semi-active schedule and the best makespan is retained.

The integration test checks that this independent exhaustive optimum equals the CP-SAT optimum on tiny fixtures.

## Development run

A locally executed seed-42 development run used:

```text
jobs                    4
machines                4
training instances/epoch 32
PPO epochs              16
PPO updates/epoch        3
hidden dimension        64
Transformer layers       2
held-out test instances 32
```

The best validation greedy makespan reached:

```text
34.781
```

Held-out test result:

```text
method            mean makespan    median

PPO greedy            38.188        38.5
SPT                   57.406        56.5
MWKR                  38.906        39.5
Earliest-start        37.219        37.0
```

Interpretation:

- PPO substantially beat SPT in this development fixture;
- PPO was slightly better than MWKR;
- PPO did **not** beat the Earliest-start heuristic;
- no statistical or general solver-superiority claim is made from one training seed.

This is retained deliberately. The project demonstrates a real RL scheduling pipeline; it does not manufacture an RL advantage when the measured benchmark does not support one.

The local runtime did not contain Gymnasium or OR-Tools, so the two package-level integration tests were skipped locally. The GitHub Actions workflow installs both packages and is designed to run the real Gymnasium checker, CP-SAT/exhaustive-oracle cross-check and a short PPO + exact-evaluation smoke experiment before the repository is considered fully validated.

## Validated GitHub Actions run

GitHub Actions run `33113397885` completed the real integration pipeline on Ubuntu 24.04 / CPython 3.12.14 with:

```text
PyTorch      2.13.0+cpu
NumPy        2.5.2
Gymnasium    1.3.0
OR-Tools     9.15.6755
```

All **10 regression/integration tests** passed, including the real Gymnasium environment checker and the CP-SAT vs independent exhaustive-oracle equality test.

The CI smoke configuration used 3 jobs × 3 machines, 12 training instances per epoch, 5 PPO epochs, 6 held-out test instances and exact CP-SAT evaluation on the first 4 test instances. The best validation greedy makespan was `31.438`.

Held-out smoke benchmark:

```text
method            mean makespan
PPO greedy            30.667
SPT                   34.000
MWKR                  26.500
Earliest-start        24.500
```

CP-SAT proved all 4 exact-evaluation instances optimal. Exact-gap results on those 4 instances were:

```text
method            mean makespan    mean exact gap    feasible rate
PPO greedy            31.250          37.315%            1.000
SPT                   32.250          45.149%            1.000
MWKR                  23.500           4.250%            1.000
Earliest-start        23.000           2.440%            1.000
Random                32.250          43.161%            1.000
```

The short CI-trained PPO policy therefore beat SPT but did **not** beat MWKR or Earliest-start. This is reported as a mechanics/integration validation, not as an RL superiority result. The important CI result is that the complete chain executed successfully:

```text
Gymnasium JSSP environment
→ masked Transformer PPO training
→ feasibility-audited schedules
→ CP-SAT optimal solutions
→ exact optimality-gap reporting
```

## Schedule feasibility audit

Every completed schedule can be checked for:

- exactly one record per operation;
- correct machine assignment;
- correct processing duration;
- nonnegative start times;
- job precedence;
- machine non-overlap.

A zero maximum violation is required in the regression suite.

## Tests

The repository currently contains ten tests covering:

- deterministic instance generation;
- valid permutation machine routes;
- exact reward telescoping;
- rejection of completed-job actions;
- tiny exhaustive optimality oracle;
- feasibility of SPT/MWKR/Earliest-start/random schedules;
- action-mask enforcement in the neural policy;
- PyTorch gradient flow;
- actual PPO parameter updates;
- `gamma != 1` rejection;
- Gymnasium environment checker integration;
- OR-Tools CP-SAT vs independent exhaustive oracle.

The final two integration tests are conditionally skipped only when those optional runtime packages are absent.

## Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python train_jssp_ppo.py --self-test
```

Regression tests:

```bash
python -m unittest discover -s tests -v
```

Development training:

```bash
python train_jssp_ppo.py \
  --seed 42 \
  --jobs 4 \
  --machines 4 \
  --rollout-instances 32 \
  --epochs 16 \
  --ppo-updates 3 \
  --minibatch-size 128 \
  --hidden-dim 64 \
  --heads 4 \
  --layers 2 \
  --test-instances 32
```

Train and evaluate against exact CP-SAT on the first eight test instances:

```bash
python train_jssp_ppo.py \
  --seed 42 \
  --jobs 4 \
  --machines 4 \
  --rollout-instances 48 \
  --epochs 40 \
  --test-instances 32 \
  --cpsat \
  --cpsat-instances 8 \
  --cpsat-time-limit 10
```

Save a checkpoint:

```bash
python train_jssp_ppo.py --checkpoint checkpoints/jssp_ppo.pt
```

## Scope

Exact statements:

- all generated schedules are audited for classical JSSP feasibility;
- raw episode return at `gamma=1` equals negative makespan;
- the tiny enumerator is exhaustive over the declared serial schedule-generation space;
- CP-SAT gaps are labeled exact only when the solver status is `OPTIMAL`.

Not claimed:

- PPO is globally optimal;
- the learned policy dominates classical dispatch rules;
- a short synthetic training run transfers to a real factory;
- CP-SAT will prove optimality for arbitrary large JSSP instances under a short time limit;
- the current environment models breakdowns, setup times, release dates, blocking, transport or stochastic processing times.

Those are natural industrial extensions, but they are outside the declared benchmark.

## References

- Schulman et al. — *Proximal Policy Optimization Algorithms*, 2017.
- Google OR-Tools — Job Shop Scheduling / CP-SAT documentation.
- Gymnasium — custom environment and environment-checker documentation.
