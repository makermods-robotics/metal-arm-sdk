"""Product identity: names, shipped profile, Star mapping and URDF of this arm. The rest of the
package reads them from here and never names the arm itself."""

from pathlib import Path
from typing import Optional

PACKAGE = "metal_arm"
DIST_NAME = "metal-arm"
CLI_NAME = "metal-arm"
DISPLAY_NAME = "Metal Arm"
MOTOR_VENDOR = "Damiao"

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = PROFILE_DIR / "metal_arm_v1.yaml"
DEFAULT_STAR_MAPPING = PROFILE_DIR / "star_to_metal_v1.json"
URDF: Optional[Path] = Path(__file__).resolve().parent / "urdf" / "metal_with_gripper.urdf"

ZERO_POSE = "folded at rest, as the arm sits unpowered, gripper fully closed"
