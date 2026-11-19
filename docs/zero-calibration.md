# Zero calibration

## The zero pose

Folded at rest, as the arm sits unpowered, with the gripper fully closed. Zeroing is the only end-user
calibration: limits, gains and the Star mapping are model data in the shipped profile.

## Commands

```bash
metal-arm zero --all                         # every follower motor
metal-arm zero --all --star-port PORT        # the follower, then the Star leader in the same pose
metal-arm zero --leader-only --star-port PORT
metal-arm zero --motor 4                     # one replaced motor
```

Torque stays off for the whole procedure: move the arm to the zero pose by hand and confirm. The
command prints the positions before and after. The zero is stored in the motors (Damiao set-zero,
`0xFE`) and in the leader servos. Without a terminal, pass `--yes-i-am-at-the-zero-pose`.

## After a power cycle

The zero survives a power cycle. Any reading outside the joint limits at enable is treated as a
stale zero and `enable()` refuses it.
