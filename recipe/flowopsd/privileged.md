# FlowSD Privileged Context

This document describes the privileged-information path implemented in
`recipe/flowopsd`. The teacher can observe information that is unavailable to
the student policy `pi_theta(y | x)` at inference time.

## Source of privileged information

By default, the privileged context is a successful response sampled for the same
problem and rollout group. It is inserted into the teacher prompt as a `Correct
solution` demonstration. The default configuration does not require a human
worked solution, an external stronger model, or environment feedback.

A rollout is successful when its sequence reward is at least
`success_reward_threshold`, which defaults to `0.5`. With binary rewards this
normally selects reward-1 trajectories.

## Demonstration selection

For each sample, the trainer selects a successful response with the same `uid`.
With `dont_reprompt_on_self_success=true`, a successful sample cannot use itself
as its own demonstration; another successful trajectory is preferred. If no
eligible demonstration exists, the sample has no privileged solution.

## Teacher prompt

The default template is:

```text
{prompt}{solution}{feedback}

Correctly solve the original question.
```

The solution block is:

```text
Correct solution:

{successful_previous_attempt}
```

The sampled response is appended after the teacher prompt, so the reference and
teacher score exactly the same response under different contexts:

```text
ref_log_prob     = log pi_ref(y | x)
teacher_log_prob = log pi_teacher(y | x, c)
```

The teacher and reference may use the same frozen parameter checkpoint. The
teacher signal comes from the privileged context `c`, not necessarily from a
larger model.

## Masking and grouping

`self_distillation_mask` indicates whether a sample received usable privileged
context. With `gate_no_context=drop`, samples without a demonstration or feedback
are excluded from the FlowSD target.

The Monte Carlo partition estimate must not combine samples that use different
privileged contexts. The trainer therefore groups by both problem identity and a
stable privileged-context key rather than by `uid` alone.

## Target energy

The sampled-token teacher gain is

```text
G_q(y; x, c) = log pi_teacher(y | x, c) - log pi_ref(y | x)
```

FlowSD combines this gain with verifier reward `R` in an energy of the form

```text
p*(y | x, c) proportional to
pi_ref(y | x) * exp(beta_q * G_q(y; x, c) + eta_R * R(y; x)).
```

## Stop-gradient boundary

The privileged side is target-side only:

- teacher and reference log probabilities are computed without gradients;
- teacher gain, reward, partition estimates, and flow-gap targets are detached;
- actor gradients flow only through the current student log probability.

Privileged information therefore changes the target distribution without being
available to the student at inference time.
