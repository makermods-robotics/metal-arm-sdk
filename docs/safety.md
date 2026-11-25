# Safety

## Before enabling

- Support the arm before enabling or releasing torque: the motors have no brakes and the arm drops
  when torque goes off away from rest.
- Keep people and obstacles outside the whole workspace. Start new tests slow and small.
- `enable()` refuses a joint reading outside its limits (a stale or missing zero), a latched motor
  fault, and a gripper at or above 90 °C (the hotter of its MOS and rotor sensors).

## Stopping and faults

- `stop()` (and every CLI command on exit) returns the joints to the pose they were enabled at, at
  30 deg/s. With `rest_window_deg` set in the profile (the Metal profile leaves it off) and the arm enabled
  that close to its rest pose (the profile's rest_pose, else zero, clipped into the limits), the joints that
  carry its weight go a little further toward that pose instead: at most 3 deg and 1.75 N*m of proportional
  effort, inside the limits. Once the return is verified, the control loop stops, each returned joint is read
  once more, and torque goes off only if every reading is fresh and no joint moved away from rest. It is not
  released when no joint returns to rest, or when a weight-carrying joint that does not return stands away from
  rest. Otherwise (and on a second Ctrl-C) the arm keeps holding its pose at full gains
  and the CLI waits for you to type `RELEASE`; without a terminal, or after a second Ctrl-C, it
  exits leaving the arm holding (exit status 1, `"then": "holding"`) and says how to release it.
- `enable()` refuses motors that already have torque on, and, on firmware that cannot report torque state,
  motors that read more than 0.1 N*m of torque (another program holding the arm); `attach()` and `rest` take
  such an arm over where it holds: no zero-gain frame and no enable reach those motors, only holding frames at the
  measured pose.
- A fault while floating (`float_mode()`, `gravity`, an arm leader), or a failing gravity model,
  stiffens the arm at its measured pose with the profile's gains and holds it as on any fault.
- On a fault (stale feedback for 0.3 s, a motor status code, the gripper at 95 °C) the arm holds
  its measured pose; the gripper is released instead. Fault evidence stays until you clear it.

## When the process dies

Joints 1-6 run with the Damiao watchdog off: if the process is killed they keep holding their last
command until power is removed. The gripper arms a 500 ms watchdog at enable (in RAM, never
saved), so it reports "communication lost" and releases on its own. `metal-arm rest` takes an arm left
energized over, brings it to rest and turns torque off; `metal-arm release` drops it where it is.

## Power and emergency stop

Use a hardware emergency stop that cuts motor power. Never rely on software holding as the only
support for a payload.
