"""trajectory.py"""

import sys
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

import robotis_gz_sim.dsrpy.dsrbind as _dsrb
from .dsenum import TrajState

__all__ = [
    "TrajGenerator",
]


class TrajGenerator(_dsrb.TrajGenerator):
    """Python wrapper class for the underlying C++ TrajGenerator."""

    # (6,) current angles [deg]
    angles: npt.NDArray[np.float64]

    # (6,) current angular velocities [deg/s]
    angvels: npt.NDArray[np.float64]

    # (6,) current angular accelerations [deg/s²]
    angaccs: npt.NDArray[np.float64]

    # (4, 4) current transformation matrix [m]
    tmat: npt.NDArray[np.float64]

    # (6, 6) current Jacobian matrix
    jmat: npt.NDArray[np.float64]

    # (6,) current screw velocity [m/s, rad/s]
    a: npt.NDArray[np.float64]

    # (6,) current screw acceleration [m/s, rad/s]
    a_d1: npt.NDArray[np.float64]

    def __init__(
        self,
        start_angles: npt.NDArray[np.generic] = np.zeros(6),
        start_angvels: npt.NDArray[np.generic] = np.zeros(6),
        start_angaccs: npt.NDArray[np.generic] = np.zeros(6),
    ) -> None:
        """
        Initialize the trajectory generator with initial joint state.

        Args:
            start_angles: (6,) start joint angles [deg]
            start_angvels: (6,) start joint velocities [deg/s]
            start_angaccs: (6,) start joint accelerations [deg/s²]
        """
        super().__init__(start_angles, start_angvels, start_angaccs)

    ####################################################################

    def update(
        self,
        dt: float,
    ) -> None:
        """
        Advance the trajectory generator state by a given timestep.

        Args:
            dt: Time step [seconds]
        """
        super().update(dt)

    ####################################################################

    def set_tcp_tmat(
        self,
        new_tcp_tmat: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the desired transformation matrix for the tool center point (TCP).

        Args:
            new_tcp_tmat: (4, 4) tcp transformation matrix [m]
        """
        super().set_tcp_tmat(new_tcp_tmat)

    def get_tcp_tmat(self) -> npt.NDArray[np.float64]:
        """
        Get the current transformation matrix of the tool center point (TCP).

        Returns:
            (4, 4) current tcp transformation matrix [m]
        """
        return super().get_tcp_tmat()

    ####################################################################

    def stop(self) -> None:
        """
        Generate and start a stopping trajectory.

        This method commands the trajectory generator to smoothly decelerate
        the current motion to a complete stop, respecting configured velocity
        and acceleration limits. The generated stop trajectory replaces any
        ongoing motion and returns immediately without waiting for completion.

        Returns:
            None

        Notes:
            - This call **does not** wait for the stop to finish.
            - Use `stop_mwait` if you need to block until the robot has fully stopped.
        """
        super().stop()

    def trapj(
        self,
        goal_angles: npt.NDArray[np.generic],
        goal_angvels: npt.NDArray[np.generic] = np.zeros(6),
        peak_angvels: npt.NDArray[np.generic] = np.array(
            [120, 120, 180, 225, 225, 255]
        ),
        peak_angaccs: npt.NDArray[np.generic] = np.array(
            [120, 120, 180, 225, 225, 255]
        ),
        duration: Optional[float] = None,
    ) -> bool:
        """
        Generate and start a joint-space trapezoidal-velocity (trapj) trajectory.

        The trajectory consists of acceleration, constant-velocity, and deceleration
        phases while respecting per-joint peak velocity/acceleration limits. If
        `duration` is None, the shortest feasible duration is computed. This method
        starts the motion and returns immediately without waiting for completion.

        Args:
            goal_angles (ndarray, shape: (6,), dtype=np.generic):
                Target joint angles [deg].

            goal_angvels (ndarray, shape: (6,), dtype=np.generic, default=zeros):
                Target joint velocities at the goal [deg/s]. Typically zeros.

            peak_angvels (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum velocities [deg/s]. Must be strictly positive.
                (Upper bounds is [120,120,180,225,225,255].)

            peak_angaccs (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum accelerations [deg/s²]. Must be strictly positive.
                (Upper bounds is [1200,1200,1800,2250,2250,2550].)

            duration (float, optional):
                Total trajectory duration [s]. Must be > 0 if provided.
                If None, the minimum feasible duration is used.

        Returns:
            bool:
                True if the trajectory was successfully generated and started;
                False if inputs are invalid or the system is in an invalid state.

        Notes:
            - This call **does not** wait for motion completion.
            - Use `trapj_mwait` if you need to block until the goal is reached.
        """
        return super().trapj(
            goal_angles, goal_angvels, peak_angvels, peak_angaccs, duration
        )

    def attrj(
        self,
        goal_angles: npt.NDArray[np.generic],
        kp: npt.NDArray[np.generic] = 10 * np.ones(6),
        goal_angvels: npt.NDArray[np.generic] = np.zeros(6),
        peak_angvels: npt.NDArray[np.generic] = np.array(
            [120, 120, 180, 225, 225, 255]
        ),
        peak_angaccs: npt.NDArray[np.generic] = np.array(
            [120, 120, 180, 225, 225, 255]
        ),
    ) -> bool:
        """
        Generate and start an attractor-based joint-space (attrj) trajectory.

        The motion is generated using an attractor control law with proportional
        gains (`kp`) in joint space, while respecting per-joint velocity and
        acceleration limits. This method starts the motion and returns immediately
        without waiting for completion.

        Args:
            goal_angles (ndarray, shape: (6,), dtype=np.generic):
                Target joint angles [deg].

            kp (ndarray, shape: (6,), dtype=np.generic, default=10*ones(6)):
                Per-joint proportional gains for the attractor controller.

            goal_angvels (ndarray, shape: (6,), dtype=np.generic, default=zeros):
                Target joint velocities at the goal [deg/s].

            peak_angvels (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum velocities [deg/s]. Must be strictly positive.
                (Upper bounds is [120,120,180,225,225,255].)

            peak_angaccs (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum accelerations [deg/s²]. Must be strictly positive.
                (Upper bounds is [1200,1200,1800,2250,2250,2550].)

        Returns:
            bool:
                True if the trajectory was successfully generated and started;
                False if inputs are invalid or the system is in an invalid state.

        Notes:
            - This call **does not** wait for motion completion.
            - Use `attrj_mwait` if you need to block until the goal is reached.
        """
        return super().attrj(
            goal_angles, kp, goal_angvels, peak_angvels, peak_angaccs
        )

    def attrl(
        self,
        goal_tmat: npt.NDArray[np.generic],
        kp: float = 50,
        goal_a: npt.NDArray[np.generic] = np.zeros(6),
        peak_endvel: float = 0.5,
        peak_endangvel: float = 180,
        peak_endacc: float = sys.float_info.max,
        peak_endangacc: float = sys.float_info.max,
    ) -> bool:
        """
        Generate and start an attractor-based Cartesian-space (attrl) trajectory.

        The motion is generated using an attractor control law with proportional
        gain (`kp`) in task space (Cartesian), while respecting linear/angular
        velocity and acceleration limits. This method starts the motion and
        returns immediately without waiting for completion.

        Args:
            goal_tmat (ndarray, shape: (4,4), dtype=np.generic):
                Target end-effector homogeneous transformation matrix.

            kp (float, default=50):
                Proportional gain for the attractor controller in Cartesian space.

            goal_a (ndarray, shape: (6,), dtype=np.generic, default=zeros):
                Target end-effector accelerations [m/s², deg/s²].

            peak_endvel (float, default=0.5):
                Maximum linear velocity [m/s]. Must be > 0.

            peak_endangvel (float, default=180):
                Maximum angular velocity [deg/s]. Must be > 0.

            peak_endacc (float, default=sys.float_info.max):
                Maximum linear acceleration [m/s²]. Must be > 0.

            peak_endangacc (float, default=sys.float_info.max):
                Maximum angular acceleration [deg/s²]. Must be > 0.

        Returns:
            bool:
                True if the trajectory was successfully generated and started;
                False if inputs are invalid or the system is in an invalid state.

        Notes:
            - This call **does not** wait for motion completion.
            - Use `attrl_mwait` if you need to block until the goal is reached.
        """
        return super().attrl(
            goal_tmat,
            kp,
            goal_a,
            peak_endvel,
            peak_endangvel,
            peak_endacc,
            peak_endangacc,
        )

    def playj(
        self,
        goal_angles_set: npt.NDArray[np.generic],
        goal_angvels_set: Optional[npt.NDArray[np.generic]] = None,
        goal_angaccs_set: Optional[npt.NDArray[np.generic]] = None,
        peak_angvels: npt.NDArray[np.generic] = np.array(
            [120, 120, 180, 225, 225, 255]
        ),
        peak_angaccs: npt.NDArray[np.generic] = np.array(
            [1200, 1200, 1800, 2250, 2250, 2550]
        ),
    ) -> bool:
        """
        Generate and start a joint-space playback (playj) trajectory.

        This method plays a sequence of joint-space waypoints, optionally with
        per-waypoint target velocities and accelerations, while respecting
        per-joint peak velocity and acceleration limits. The motion is started
        and this call returns immediately without waiting for completion.

        Args:
            goal_angles_set (ndarray, shape: (N, 6), dtype=np.generic):
                Sequence of target joint angles [deg] for each waypoint.
                N is the number of waypoints.

            goal_angvels_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint velocities [deg/s] for each waypoint.
                If None, zeros are assumed for all waypoints.

            goal_angaccs_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint accelerations [deg/s²] for each waypoint.
                If None, zeros are assumed for all waypoints.

            peak_angvels (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum velocities [deg/s]. Must be strictly positive.

            peak_angaccs (ndarray, shape: (6,), dtype=np.generic, default=[1200,1200,1800,2250,2250,2550]):
                Per-joint maximum accelerations [deg/s²]. Must be strictly positive.

        Returns:
            bool:
                True if the trajectory was successfully generated and started;
                False if inputs are invalid or the system is in an invalid state.

        Notes:
            - `goal_angles_set`, `goal_angvels_set` (if provided), and
              `goal_angaccs_set` (if provided) must all have the same first
              dimension N (number of waypoints).
            - This call **does not** wait for motion completion.
            - If any per-joint limits are violated, the command may be rejected.
        """
        return super().playj(
            goal_angles_set,
            goal_angvels_set,
            goal_angaccs_set,
            peak_angvels,
            peak_angaccs,
        )

    ####################################################################

    def traj_state(self) -> TrajState:
        """
        Get the current trajectory generation state.

        This method returns the internal state of the trajectory generator.

        Returns:
            TrajState:
                Current trajectory generator state. One of:

                - `TrajState.STOP`: No trajectory generation is active.
                - `TrajState.STOPPING`: Stopping trajectory is running.
                - `TrajState.TRAPJ`: Trapezoidal joint-space trajectory is running.
                - `TrajState.ATTRJ`: Attractor-based joint-space trajectory is running.
                - `TrajState.ATTRL`: Attractor-based task-space trajectory is running.
        """
        return TrajState(super().traj_state())

    ####################################################################

    def goal_angles(self) -> Optional[npt.NDArray[np.float64]]:
        """
        Get the goal joint angles of the currently active trajectory.

        This method returns the target joint angles (shape: (6,)) for the current
        joint-space trajectory, if one is active. If no trajectory is active or
        the current mode is not joint-space (e.g., task-space trajectory), it returns None.

        Returns:
            Optional[np.ndarray]:
                - (6,) array of goal joint angles [deg], or
                - None if goal_angvels cannot be retrieved.
        """
        return super().goal_angles()

    def goal_angvels(self) -> Optional[npt.NDArray[np.float64]]:
        """
        Get the goal joint velocities of the currently active trajectory.

        This method returns the target joint velocities (shape: (6,)) for the current
        joint-space trajectory, if one is active. If no trajectory is active or
        the current mode is not joint-space (e.g., task-space trajectory), it returns None.

        Returns:
            Optional[np.ndarray]:
                - (6,) array of goal joint velocities [deg/s], or
                - None if goal_angvels cannot be retrieved.
        """
        return super().goal_angvels()

    def goal_tmat(self) -> Optional[npt.NDArray[np.float64]]:
        """
        Get the goal end-effector pose of the currently active trajectory.

        This method returns the target transformation matrix (shape: (4, 4)) for the
        current task-space trajectory, if one is active. If no trajectory is active
        or goal_tmat cannot be retrieved, it returns None.

        Returns:
            Optional[np.ndarray]:
                - (4, 4) array representing the goal SE(3) pose, or
                - None if goal_tmat cannot be retrieved.
        """
        return super().goal_tmat()

    def goal_a(self) -> Optional[npt.NDArray[np.float64]]:
        """
        Get the goal end-effector screw velocity of the currently active trajectory.

        This method returns the target screw velocity (shape: (6,)) for the current
        joint-space trajectory, if one is active. If no trajectory is active
        or goal_tmat cannot be retrieved, it returns None.

        Returns:
            Optional[np.ndarray]:
                - (6,) array of goal end-effector screw velocities [m/s, rad/s], or
                - None if goal_tmat cannot be retrieved.
        """
        return super().goal_a()

    def goal_reached(
        self,
        angles_enorm_thold: Optional[float] = 2,
        pos_enorm_thold: Optional[float] = 0.002,
        rot_enorm_thold: Optional[float] = 3,
        angvels_enorm_thold: Optional[float] = 4,
        vel_enorm_thold: Optional[float] = 0.004,
        w_enorm_thold: Optional[float] = 6,
    ) -> bool:
        """
        Check whether the trajectory has reached its goal within specified thresholds.

        This method returns True if the trajectory has reached its target state within
        the given error norms for joint angles, end-effector position and rotation,
        and optionally for velocities. Returns False if still moving toward the goal.
        Returns None if there is no active trajectory to evaluate.

        Args:
            angles_enorm_thold (Optional[float], default=2):
                Threshold for joint angle error norm [deg].
                If None or moving with attrl, this check is skipped.

            pos_enorm_thold (Optional[float], default=0.002):
                Threshold for end-effector position error norm [m].
                If None, this check is skipped.

            rot_enorm_thold (Optional[float], default=3):
                Threshold for end-effector rotation error norm [deg].
                If None, this check is skipped.

            angvels_enorm_thold (Optional[float], default=4):
                Threshold for joint velocity error norm [deg/s].
                If None or moving with attrl, this check is skipped.

            vel_enorm_thold (Optional[float], default=0.004):
                Threshold for end-effector linear velocity error norm [m/s].
                If None, this check is skipped.

            w_enorm_thold (Optional[float], default=6):
                Threshold for end-effector angular velocity error norm [deg/s].
                If None, this check is skipped.

        Returns:
            bool:
                - True: Goal has been reached within specified tolerances.
                - False: Still moving toward the goal.
        """
        return super().goal_reached(
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
        )

    ####################################################################

    def solve_forward(
        self,
        angles: npt.NDArray[np.generic],
    ) -> npt.NDArray[np.float64]:
        """
        Compute the forward kinematics for the given joint angles.

        This method computes the end-effector pose corresponding to the
        specified joint configuration using the current kinematic model.

        Args:
            angles (np.ndarray):
                Joint angles [deg], shape (6,).
                The input joint configuration for which to compute the pose.

        Returns:
            np.ndarray:
                (4, 4) transformation matrix [SE(3)] representing the
                end-effector pose in the base frame.
        """
        return super().solve_forward(angles)

    def solve_inverse(
        self,
        initial_angles: npt.NDArray[np.generic],
        target_tmat: npt.NDArray[np.generic],
        step_max: float = 10000,
        enorm_threshold: float = 1e-4,
        damping: float = 0.01,
        base_link_tmat: npt.NDArray[np.generic] = np.eye(4),
    ) -> Tuple[npt.NDArray[np.float64], bool]:
        """
        Compute the inverse kinematics for the given target transformation matrix.

        This method computes the joint angles that achieve the specified end-effector
        pose using the current kinematic model.

        Args:
            initial_angles (np.ndarray):
                Initial joint angles [deg], shape (6,).
                The starting point for the inverse kinematics search.

            target_tmat (np.ndarray):
                Target end-effector pose [SE(3)], shape (4, 4).
                The desired pose to achieve.

            step_max (int):
                Maximum number of iterations for the inverse kinematics solver.

            enorm_threshold (float):
                Error norm threshold for convergence.

            damping (float):
                Damping factor to improve numerical stability.

            base_link_tmat (np.ndarray):
                Base link transformation matrix [SE(3)], shape (4, 4).

        Returns:
            Tuple[np.ndarray, bool]:
                - Joint angles [deg], shape (6,).
                - Convergence status.
        """
        return super().solve_inverse(
            initial_angles,
            target_tmat,
            step_max,
            enorm_threshold,
            damping,
            base_link_tmat,
        )
