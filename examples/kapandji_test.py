"""Kapandji example: move each finger pose, then sweep thumb waypoints."""

import time

import numpy as np

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand


THUMB_IDS = (0, 1, 2, 3)
FINGER_IDS = {
    "index": (4, 5, 6),
    "middle": (7, 8, 9),
    "ring": (10, 11, 12),
}

# Per finger: PIP, MCP flex, MCP abduct.
FINGER_TARGET_RAD = np.array([-0.51388356, -0.95322344, 0.0])
THUMB_TARGETS_RAD = {
    "index": np.array(
        [
            [-1.18576715, -0.01227184, -0.32060198, 0.59211658],
            [-0.7823302, -0.13805827, 0.06902914, 1.14588365],
            [-0.96333994, -0.40957287, 0.42491268, 1.10139821],
        ]
    ),
    "middle": np.array(
        [
            [-0.21629129, -0.69489329, -0.31293208, 1.11367005],
            [-0.18100973, -0.65807776, 0.11965051, 1.40972834],
            [-0.1043107, -0.77772826, 0.48166997, 1.44040796],
        ]
    ),
    "ring": np.array(
        [
            [-0.06902914, -0.89124284, 0.19634955, 1.98957308],
            [-0.55223309, -0.41570879, 0.35434957, 1.93895172],
            [-0.34667966, -0.61052435, 0.70869913, 1.8208352],
        ]
    ),
}


def load_config() -> HandConfig:
    if DEFAULT_CONFIG_PATH.exists():
        return HandConfig.load(DEFAULT_CONFIG_PATH)
    return HandConfig()


def set_motor_targets(
    hand: MidasHand,
    target: np.ndarray,
    motor_ids: tuple[int, ...],
    values: np.ndarray,
) -> None:
    for motor_id, value in zip(motor_ids, values):
        if motor_id not in hand.motor_ids:
            raise ValueError(f"Motor ID {motor_id} is not configured")
        target[hand.motor_ids.index(motor_id)] = value


def command_and_wait(hand: MidasHand, target: np.ndarray) -> None:
    hand.set_positions(target, clip=True)
    if not hand.wait_until_reached(
        target,
        tolerance_rad=0.1,
        velocity_threshold_rad_s=0.05,
        timeout_s=6.0,
        poll_interval_s=0.01,
    ):
        raise TimeoutError("Hand did not reach target within 6.00s")
    print(f"Motor positions rad: {hand.read_pos()}")


def run_finger_sequence(
    hand: MidasHand,
    finger_name: str,
    target: np.ndarray,
    previous_finger_name: str | None = None,
) -> None:
    finger_ids = FINGER_IDS[finger_name]
    if previous_finger_name is not None:
        previous_finger_ids = FINGER_IDS[previous_finger_name]
        set_motor_targets(hand, target, previous_finger_ids, np.zeros(3))
        hand.set_positions(target, clip=True)
        time.sleep(0.1)

    set_motor_targets(hand, target, finger_ids, FINGER_TARGET_RAD)
    for waypoint_idx, thumb_target in enumerate(THUMB_TARGETS_RAD[finger_name]):
        set_motor_targets(hand, target, THUMB_IDS, thumb_target)
        if waypoint_idx == 0:
            if previous_finger_name is None:
                print(
                    f"Moving {finger_name} finger to Kapandji target while "
                    "thumb starts its first waypoint."
                )
            else:
                print(
                    f"Moving {previous_finger_name} finger to zero and "
                    f"{finger_name} finger to Kapandji target while thumb "
                    "starts its first waypoint."
                )
        else:
            print(f"Moving thumb for {finger_name}: {thumb_target}")
        command_and_wait(hand, target)
        time.sleep(0.1)


def main() -> None:
    hand = MidasHand(load_config())
    try:
        print(f"Connected on {hand.port}")
        hand.configure(enable_torque=False)
        hand.enable_torque()

        hand.set_motion_profile(
            profile_velocity_rad_s=3.0,
            profile_acceleration_rad_s2=20.0,
            motor_ids=hand.motor_ids,
        )

        time.sleep(1.0)
        while True:
            target = np.zeros(len(hand.motor_ids))
            previous_finger_name = None
            for finger_name in ("index", "middle", "ring"):
                run_finger_sequence(hand, finger_name, target, previous_finger_name)
                previous_finger_name = finger_name

            print("Returning all motors to software zero.")
            command_and_wait(hand, np.zeros(len(hand.motor_ids)))
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("Interrupted; disabling torque.")
    finally:
        print("Disabling torque. Please wait...")
        hand.shutdown()


if __name__ == "__main__":
    main()
