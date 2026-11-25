# Damiao wire protocol

Classic CAN, 1 Mbit/s, standard 11-bit ids, 8-byte frames. Code: `metal_arm/protocol.py` (codec),
`metal_arm/motor.py` (transactions).

## Ids

| | Metal Arm |
|---|---|
| send id (`ESC_ID`, register 8) | 1..7, joint order shoulder_pan .. gripper |
| master / reply id (`MST_ID`, register 7) | 0x11..0x17 (`host_id` 0x10 + send id) |
| shared request id | 0x7FF (registers, refresh) |

The SDK accepts send ids 1..15: a state frame carries only the low nibble of the id.

## Host to motor

| Frame | CAN id | Payload |
|---|---|---|
| enable | send id | `FF FF FF FF FF FF FF FC` |
| disable | send id | `FF ×7 FD` |
| set zero (persists) | send id | `FF ×7 FE` |
| clear error | send id | `FF ×7 FB` |
| MIT command | send id | see below |
| refresh (read-only state request) | 0x7FF | `id_lo id_hi CC 00 00 00 00 00` |
| register read | 0x7FF | `id_lo id_hi 33 rid 00 00 00 00` |
| register write (RAM) | 0x7FF | `id_lo id_hi 55 rid v0 v1 v2 v3` (little-endian u32 or float32) |
| save all registers to flash | 0x7FF | `id_lo id_hi AA 01 00 00 00 00` |

Commands go to send id + a mode offset (MIT 0x000, pos-vel 0x100, vel 0x200, force-position
0x300); the SDK only drives MIT mode.

MIT command, big-endian bit fields: `q[16] dq[12] kp[12] kd[12] tau[12]`, each value clamped to its
range and mapped linearly with truncation (`int((x - lo) / (hi - lo) * (2^bits - 1))`):
q ±PMAX rad, dq ±VMAX rad/s, kp 0..500, kd 0..5, tau ±TMAX N·m. Motor output torque is
`kp·(q_cmd − q) + kd·(dq_cmd − dq) + tau`. A payload whose first seven bytes would all be 0xFF
(and so read as a command) has its position lowered by one LSB.

| model | joints | PMAX rad | VMAX rad/s | TMAX N·m |
|---|---|---|---|---|
| `jlo` | 1, 3, 4 | 6.28 | 10 | 30 |
| `j2` | 2 | 6.28 | 10 | 120 |
| `jhi` | 5, 6, 7 | 6.28 | 30 | 20 |

These must equal the motor's own PMAX/VMAX/TMAX registers (21/22/23); `Motor.validate()` checks
them and CTRL_MODE (10) == 1 (MIT) at connect.

## Motor to host (on the master id)

State frame, the reply to every command, MIT frame and refresh:

| byte | content |
|---|---|
| 0 | high nibble status, low nibble send id |
| 1-2 | q (16 bit) |
| 3, 4 high nibble | dq (12 bit) |
| 4 low nibble, 5 | tau (12 bit) |
| 6 | MOS temperature, °C |
| 7 | rotor temperature, °C |

Status: 0 disabled, 1 enabled, 8 overvoltage, 9 undervoltage, A overcurrent, B MOS
overtemperature, C rotor overtemperature, D communication lost (watchdog), E overload. Any status
other than 0/1 is a fault; the SDK latches it until `clear_fault()` (0xFB). The refresh never
clears anything.

Register reply: echoes the request header `id_lo id_hi op rid` then the value. A write echoes
the written value. A disabled state frame whose position bytes are `00 33|55|AA` is
indistinguishable from a register reply and is treated as one (that sample is dropped).

## Registers used

| rid | name | type | use |
|---|---|---|---|
| 7 | MST_ID | u32 | id change |
| 8 | ESC_ID | u32 | id change |
| 9 | TIMEOUT | u32 | watchdog, 50 µs ticks (500 ms = 10000); 0 = off |
| 10 | CTRL_MODE | u32 | 1 MIT, 2 pos-vel, 3 vel, 4 force-position |
| 21-23 | PMAX, VMAX, TMAX | float32 | MIT scaling check |

u32 registers: 7-10, 13-16, 35, 36; all others float32 (Damiao DM_CAN). `metal-arm params` lists
the full map (`Motor.PARAMS`).

`arm_watchdog()` writes TIMEOUT in RAM only (never saved), reads it back, and `restore_watchdog()`
writes the previous value back. On the Metal Arm only the gripper runs a watchdog (500 ms); joints
1-6 keep 0 so a lost host never drops the arm.

## Late replies (the gripper)

Motor 7 often emits the reply to request N only when the next frame appears on the bus, sometimes
twice, occasionally never. Every sync query therefore matches replies by content, sends a flush
frame (a refresh of a neighbouring id) after 3 ms of silence (up to 3 per attempt), keeps the
newest state reply after one trailing flush, and re-sends the request on a miss.

## Id change (`Motor.set_id`)

1. Refresh and read ESC_ID at the new id: any answer refuses the change.
2. Refuse unless the motor reports disabled (the save is ignored while enabled).
3. Write MST_ID = 0x10 + new id, then ESC_ID = new id, through the old id. Echoes are accepted on
   the old or new master id; the ESC_ID echo may name either id or be missing.
4. Read ESC_ID through the new id, else the old id; save (0xAA) through whichever answers.
5. Verify ESC_ID and MST_ID through the new id. If the motor still answers only at the old id the
   change is saved but not applied: power-cycle and scan.

Metal motors apply MST_ID and ESC_ID writes immediately, so no power cycle is needed. The
procedure also handles firmware that applies them only after one.

`Arm.set_motor_id` (and `metal-arm set-id`) gives a joint's id in the profile only to a motor the profile does not
have, and only when the caller names that joint: a new motor on id 9 replacing the gripper is
`metal-arm set-id --arm left --from 9 --to 7 --replacing gripper`. A joint's own motor never takes another
joint's id.

## Sources

Damiao DM_CAN reference driver and motor manual (register map, 0x33/0x55/0xAA, 0xCC refresh,
status codes); LeRobot Damiao bus (makermods-robotics fork), whose MIT packing the encoder matches
byte for byte (tests/unit/test_protocol.py golden frames); MakerModsLab `metal_gripper.py` (late
replies, register echo matching, 500 ms watchdog in RAM); the vendor Metal C++ SDK
`dm_motor_writer.cpp` / `dm_motor_reader.cpp`; the Dream console (status names, comm-lost
handling, joint watchdog policy).
