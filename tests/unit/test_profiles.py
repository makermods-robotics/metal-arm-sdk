"""The shipped profile and Star mapping hold the reference values below."""

import json
import math
from pathlib import Path

import pytest
import yaml

from metal_arm import protocol
from metal_arm._product import DEFAULT_PROFILE, DEFAULT_STAR_MAPPING, URDF
from metal_arm.config import ArmConfig
from metal_arm.kinematics import resolve_urdf
from metal_arm.mapping import JointMapper

# name: (motor_id, reply_id, model, limits_deg, kp, kd)
REFERENCE = {
    "shoulder_pan": (1, 0x11, "jlo", (-160.0, 160.0), 100, 5.0),
    "shoulder_lift": (2, 0x12, "j2", (-180.0, 0.0), 100, 5.0),
    "elbow_flex": (3, 0x13, "jlo", (0.0, 180.0), 100, 5.0),
    "wrist_flex": (4, 0x14, "jlo", (-123.0, 81.0), 50, 5.0),
    "wrist_yaw": (5, 0x15, "jhi", (-85.0, 85.0), 20, 0.6),
    "wrist_roll": (6, 0x16, "jhi", (-145.0, 145.0), 20, 0.6),
    "gripper": (7, 0x17, "jhi", (0.0, 137.5), 20, 0.6),
}

STAR_DIRECTIONS = {
    "shoulder_pan": -1.0, "shoulder_lift": -1.0, "elbow_flex": -0.667, "wrist_flex": -1.0,
    "wrist_yaw": 1.0, "wrist_roll": -1.0, "gripper": 1.895,
}


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(DEFAULT_PROFILE.read_text())


def test_profile_matches_the_reference(raw):
    joints = raw["joints"]
    assert [j["name"] for j in joints] == list(REFERENCE)
    assert raw["required_motor_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert raw["host_id"] == protocol.DEFAULT_MASTER_OFFSET
    for j in joints:
        motor_id, reply_id, model, limits, kp, kd = REFERENCE[j["name"]]
        assert (j["motor_id"], j["reply_id"], j["model"], j["kp"], j["kd"]) == (motor_id, reply_id, model, kp, kd)
        assert tuple(j["limits_deg"]) == limits
        assert j["reply_id"] == raw["host_id"] + j["motor_id"]


def test_model_scaling():
    assert protocol.MODELS["jlo"] == protocol.MotorModel(6.28, 10.0, 30.0)
    assert protocol.MODELS["j2"] == protocol.MotorModel(6.28, 10.0, 120.0)
    assert protocol.MODELS["jhi"] == protocol.MotorModel(6.28, 30.0, 20.0)


def test_limits_fit_inside_the_mit_position_range(raw):
    for j in raw["joints"]:
        p_max = protocol.MODELS[j["model"]].p_max
        assert all(abs(math.radians(v)) <= p_max for v in j["limits_deg"])


def test_gains_inside_protocol_ranges(raw):
    for j in raw["joints"]:
        assert protocol.KP_MIN <= j["kp"] <= protocol.KP_MAX
        assert protocol.KD_MIN <= j["kd"] <= protocol.KD_MAX


def test_speed_cap_and_safety_settings(raw):
    assert raw["max_velocity_deg"] == 280
    *arm_joints, gripper = raw["joints"]
    assert gripper["can_timeout_ms"] == 500
    assert gripper["can_timeout_ms"] * protocol.TIMEOUT_TICKS_PER_MS == 10_000
    assert (gripper["max_temperature_c"], gripper["enable_temperature_c"]) == (95, 90)
    assert gripper["hold_torque_nm"] == 0.5 and gripper["closing_direction"] == -1
    assert gripper["release_on_fault"] and not gripper["return_to_rest"]
    for j in arm_joints:
        assert "can_timeout_ms" not in j and "max_temperature_c" not in j


def test_star_mapping_matches_the_reference():
    mapping = json.loads(DEFAULT_STAR_MAPPING.read_text())
    assert [(j["name"], j["servo"]) for j in mapping["joints"]] == [(n, i) for i, n in enumerate(STAR_DIRECTIONS)]
    for j in mapping["joints"]:
        assert j["direction"] == STAR_DIRECTIONS[j["name"]]
        lo, hi = REFERENCE[j["name"]][3]
        expected = [0, 115] if j["name"] == "gripper" else [int(lo), int(hi)]
        assert j["range_deg"] == expected


def test_urdf_is_shipped():
    assert URDF is not None and URDF.is_file()


def test_profile_loads_through_arm_config():
    cfg = ArmConfig.from_yaml(str(DEFAULT_PROFILE))
    assert [j.motor_id for j in cfg.joints] == list(range(1, 8))
    assert cfg.max_velocity == pytest.approx(math.radians(280))
    assert cfg.joints[-1].reply_id == 0x17


def test_star_mapping_loads_through_mapper():
    JointMapper.from_json(DEFAULT_STAR_MAPPING)


def test_profile_model_is_the_gravity_reference():
    """urdf, signs, TCP, caps and factors in the profile are the ones gravity_fixtures.json holds."""
    spec = json.loads((Path(__file__).resolve().parent / "gravity_fixtures.json").read_text())["chain"]
    cfg = ArmConfig.from_yaml(str(DEFAULT_PROFILE))
    arm = cfg.joints[:6]
    assert resolve_urdf(cfg.urdf) == URDF == Path(__file__).resolve().parents[2] / spec["urdf"]
    assert [math.degrees(v) for v in cfg.urdf_base_rpy] == pytest.approx(spec["base_rpy_deg"])
    assert cfg.tcp_xyz == pytest.approx(spec["tcp_xyz"])
    assert [math.degrees(v) for v in cfg.tcp_rpy] == pytest.approx(spec["tcp_rpy_deg"])
    assert [j.urdf_sign for j in arm] == spec["sign"]
    assert [math.degrees(j.urdf_offset) for j in arm] == spec["offset_deg"]
    assert [j.gravity_limit_nm for j in arm] == spec["gravity_limit_nm"]
    assert [j.gravity_factor for j in arm] == spec["gravity_factor"]
    for j in arm:
        assert j.gravity_limit_nm < protocol.MODELS[j.model].t_max


def test_leader_friction_model():
    cfg = ArmConfig.from_yaml(str(DEFAULT_PROFILE))
    assert [j.friction_viscous for j in cfg.joints] == [0.21, 0.998, 0.572, 0.294, 0.0045, 0.0105, 0.01]
    assert [j.friction_velocity_clamp for j in cfg.joints] == [math.inf, math.inf, 0.8, 1.2, math.inf, math.inf, 3.0]
    assert [j.friction_coulomb for j in cfg.joints] == [0.0] * 6 + [0.03]
    assert [j.friction_rest_nm for j in cfg.joints] == [0.0] * 6 + [0.06]
    assert cfg.friction_deadzone == 0.05


def test_gripper_fully_open_at_the_end_of_its_stroke():
    gripper = ArmConfig.from_yaml(DEFAULT_PROFILE).joints[-1]
    assert gripper.name == "gripper" and gripper.closing_direction == -1
    assert math.degrees(gripper.open_position) == pytest.approx(116.4)   # short of the 137.5 limit
