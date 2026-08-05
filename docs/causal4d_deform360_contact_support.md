# Deform360 source contact/support mechanism diagnostic

## Purpose

The frozen six-object Deform360 source backend failed before target access. A
source-only attribution showed that choosing a separate quality-constrained
physical parameter candidate for each episode was already insufficient: the
per-episode oracle beat persistence in only 6 of 28 strain-valid episodes. The
next narrow hypothesis, missing initial object velocity, was then tested on all
30 opened source episodes. Its zero baseline reproduced exactly, but both rigid
contact translation and graph-harmonic velocity initialization were slightly
worse overall and failed the predeclared win gate.

This diagnostic therefore tests the next directly observable mechanisms without
widening the velocity model or changing the registered Causal4D estimator:

- initial support height relative to the Warp ground plane;
- visual contact-patch width;
- tactile versus opening-derived contact timing; and
- a no-contact negative control.

The study is source-only. It uses no calibration outcome, target prefix, target
future geometry, or target tactile stream. A positive source result would justify
only a separately registered replication; it would not revise the frozen
Deform360 result or the locked 36-execution physical protocol.

## Locked evidence and cohort

The diagnostic verifies before execution:

- all 51 files in `deform360-replication-source-backend-v1`;
- the frozen source-backend decision;
- the completed source-failure attribution result;
- the completed negative prefix-kinematics result;
- the exact Deform360 protocol and five objects with complete source grids; and
- one fixed per-episode candidate selected from each already-opened source grid
  by the original quality-constrained source-oracle rule.

The evaluated cohort is:

| Stratum | Objects | Episodes |
| --- | --- | ---: |
| Filament | `002-rope-silk`, `081-stripe-rope` | 12 |
| Sheet | `083-blanket-cloth`, `085-scarf-cloth` | 12 |
| Volumetric | `170-spider` | 6 |

`092-squirrel` remains outside the panel because its source geometry failed
before a complete candidate grid existed. No object or episode may be removed
after observing the mechanism results.

## Fixed policies

Every policy reuses the same source-selected physical candidate, prefix graph,
future scoring frames, strain threshold, controller trajectory, and zero initial
object velocity unless a policy explicitly changes one mechanism below.

1. `registered_v1` reproduces the archived backend exactly: tactile contact
   schedule, eight nearest gripper taxels per patch, and one millimetre initial
   ground clearance.
2. `support_touching_v1` changes only the initial ground clearance from one
   millimetre to zero.
3. `support_lifted_5mm_v1` changes only the initial ground clearance from one to
   five millimetres.
4. `contact_patch_4_v1` changes only the prefix visual contact patch from eight
   to four nearest taxels.
5. `contact_patch_12_v1` changes only the prefix visual contact patch from eight
   to twelve nearest taxels.
6. `opening_contact_schedule_v1` changes only the contact-active schedule from
   the source tactile window to the already source-fitted, causal opening
   threshold. The physical contact association remains the registered
   eight-taxel association.
7. `contact_disabled_v1` uses the registered association but disables all
   controller springs. It is a negative control and can never be promoted as a
   mechanism candidate, even when it scores better.

The patch-width candidates recompute their prefix-only association from the same
prefix hull and robot state. They do not read future object geometry while
constructing the policy. The opening schedule reads only the released robot
opening trajectory and the pre-existing source-fitted threshold.

## Baseline reproduction and gate

No mechanism result is interpretable unless all 30 `registered_v1` reruns
reproduce the archived per-episode Chamfer and p99-strain values within:

- `0.5 mm` absolute mean-Chamfer tolerance; and
- `0.01` absolute p99 relative-edge-strain tolerance.

Each candidate policy is judged independently. A candidate is source-supported
only when it:

- has at least 24 common finite episodes;
- improves equal-episode mean Chamfer by at least 5%;
- wins at least 60% of common episodes while satisfying the strain limit; and
- does not reduce the strain-valid episode count.

The no-contact control is reported using the same metrics but is excluded from
the candidate set. A better no-contact result would diagnose harmful contact
realization; it would not support a physically meaningful no-contact estimator.

## Metrics and retained artifacts

The result retains, overall and by object:

- mean and late symmetric Chamfer distance;
- p99 and maximum relative edge strain;
- finite and strain-valid counts;
- equal-episode win fractions;
- quality rescues and regressions;
- source-selected candidate identities;
- tactile/opening active fractions and schedule disagreement;
- contact-node identities, patch widths, and patch-to-node distances; and
- the complete target-closed information boundary.

The workflow also publishes the exact runtime-selection record, repository
revisions, CUDA/Warp environment, result/runtime file hashes, and a SHA-256
inventory. The conditional reproduction runtime introduced by the completed
prefix-kinematics study remains in force: the recorded NumPy `2.5.1` milestone
is not rewritten, while the available NumPy `1.26.4` runtime is admissible only
when the registered baseline reproduces.

## Execution

The permanent workflow is read-only and runs the GPU panel only after an explicit
manual dispatch. Its contract job runs automatically on relevant pull requests.
Manual execution on the registered runner is:

```bash
bash scripts/remote/run_deform360_contact_support_workflow.sh \
  /path/to/Causal4D \
  /path/to/BayesianPhysTwin \
  /path/to/deform360 \
  /path/to/official-PhysTwin \
  /path/to/output
```

The exact protocol lock is
`configs/causal4d_public/deform360_contact_support_v1.json`.

## Interpretation boundary

The diagnostic may establish one of three bounded outcomes:

1. a support-height candidate passes, motivating a separately registered support
   replication;
2. a contact-patch or opening-schedule candidate passes, motivating a separately
   registered contact-realization replication; or
3. no candidate passes, closing these simple mechanisms and redirecting work to
   topology-appropriate representation or deeper within-episode dynamics.

It cannot establish target transfer, physical-object confirmation, calibrated
uncertainty, contact recovery accuracy, material-parameter identification, or a
change to the registered 36-execution method. The primary physical evidence
state remains `0/36` until genuine registered executions are acquired.
