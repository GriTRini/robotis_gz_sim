"""robot.py"""

import sys
from typing import Optional, Tuple

import datetime

import numpy as np
import numpy.typing as npt

import robotis_gz_sim.dsrpy.dsrbind as _dsrb
from .dsenum import (
    OpenConnError,
    ServoOnError,
    ServoOffError,
    CloseConnError,
    SafetyMode,
    SafetyModeEvent,
    TrajState,
    RobotState,
)


__all__ = [
    "Robot",
]


class Robot(_dsrb.Robot):
    """Robot"""

    def __init__(
        self,
        update_dt: datetime.timedelta = datetime.timedelta(milliseconds=1),
    ) -> None:
        super().__init__(update_dt)

    def open_connection(
        self,
        strIpAddr: str = "192.168.137.100",
        usPort: int = 12345,
        timeout_init_tp: datetime.timedelta = datetime.timedelta(
            milliseconds=100
        ),
        timeout_get_ctrl: datetime.timedelta = datetime.timedelta(
            milliseconds=500
        ),
    ) -> OpenConnError:
        return OpenConnError(
            super().open_connection(
                strIpAddr, usPort, timeout_init_tp, timeout_get_ctrl
            )
        )

    def connect_rt(
        self,
        strRobotIp: str = "192.168.137.100",
        usPort: int = 12347,
    ) -> bool:
        return super().connect_rt(strRobotIp, usPort)

    def servo_on(
        self,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=3000),
    ) -> ServoOnError:
        return ServoOnError(super().servo_on(timeout))

    def servo_off(
        self,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=100),
    ) -> ServoOffError:
        return ServoOffError(super().servo_off(timeout))

    def disconnect_rt(self) -> None:
        super().disconnect_rt()

    def close_connection(
        self,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=100),
    ) -> CloseConnError:
        return CloseConnError(super().close_connection(timeout))

    ###################################################

    def set_tcp_tmat(
        self,
        tcp_tmat: npt.NDArray[np.generic],
    ) -> None:
        """
        Set the desired transformation matrix for the tool center point (TCP).

        Args:
            new_tcp_tmat: (4, 4) tcp transformation matrix [m]
        """
        super().set_tcp_tmat(tcp_tmat)

    def get_tcp_tmat(self) -> npt.NDArray[np.float64]:
        """
        Get the current transformation matrix of the tool center point (TCP).

        Returns:
            (4, 4) current tcp transformation matrix [m]
        """
        return super().get_tcp_tmat()

    def set_tool(
        self,
        lpszSymbol: str,
    ) -> bool:
        return super().set_tool(lpszSymbol)

    def get_tool(self) -> str:
        return super().get_tool()

    def add_tool(
        self,
        lpszSymbol: str,
        fWeight: float,
        fCog: Tuple[float, float, float],
        fInertia: Tuple[float, float, float, float, float, float],
    ) -> bool:
        return super().add_tool(lpszSymbol, fWeight, fCog, fInertia)

    def del_tool(
        self,
        lpszSymbol: str,
    ) -> bool:
        return super().del_tool(lpszSymbol)

    def set_safety_mode(
        self,
        eSafetyMode: SafetyMode,
        eSafetyEvent: SafetyModeEvent,
    ) -> bool:
        return super().set_safety_mode(eSafetyMode.value, eSafetyEvent.value)

    def change_collision_sensitivity(
        self,
        fSensitivity: float,
    ) -> bool:
        return super().change_collision_sensitivity(fSensitivity)

    def set_servoj_target_time(
        self,
        new_target_time: float,
    ) -> None:
        super().set_servoj_target_time(new_target_time)

    def get_servoj_target_time(self) -> float:
        return super().get_servoj_target_time()

    ###################################################

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

    def stop_mwait(
        self,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=10000),
    ) -> bool:
        """
        Generate and execute a stopping trajectory, then block until the stop completes
        or the timeout is exceeded.

        This method commands the trajectory generator to smoothly decelerate
        the current motion to a complete stop, respecting configured velocity
        and acceleration limits. The method blocks until the stop motion is
        completed or the timeout expires.

        Args:
            timeout (datetime.timedelta, default=10s):
                Maximum time to wait for the stop motion to complete.

        Returns:
            bool:
                True if the stop motion completed successfully before the timeout;
                False otherwise (e.g., interrupted, thresholds not met, or timeout).

        Notes:
            - This call **blocks** until the stop is complete or the timeout occurs.
            - Use `stop` if you only want to initiate a stop without waiting.
        """
        return super().stop_mwait(timeout)

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

    def trapj_mwait(
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
        angles_enorm_thold: Optional[float] = 0,
        pos_enorm_thold: Optional[float] = None,
        rot_enorm_thold: Optional[float] = None,
        angvels_enorm_thold: Optional[float] = 0,
        vel_enorm_thold: Optional[float] = None,
        w_enorm_thold: Optional[float] = None,
    ) -> bool:
        """
        Generate and execute a joint-space trapezoidal-velocity (trapj) trajectory,
        then block until the goal is reached within given tolerances.

        The trajectory consists of acceleration, constant-velocity, and deceleration
        phases while respecting per-joint peak velocity/acceleration limits. If
        `duration` is None, the shortest feasible duration is computed.

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

            angles_enorm_thold (float, optional):
                Threshold for joint-angle error norm [deg]. If None, this check is skipped.

            pos_enorm_thold (float, optional):
                Threshold for end-effector position error norm [m]. If None, skipped.

            rot_enorm_thold (float, optional):
                Threshold for end-effector rotation error norm [deg]. If None, skipped.

            angvels_enorm_thold (float, optional):
                Threshold for joint-velocity error norm [deg/s]. If None, skipped.

            vel_enorm_thold (float, optional):
                Threshold for end-effector linear-velocity error norm [m/s]. If None, skipped.

            w_enorm_thold (float, optional):
                Threshold for end-effector angular-velocity error norm [deg/s]. If None, skipped.

        Returns:
            bool:
                True if the motion completes and the final state satisfies all
                provided thresholds; False otherwise (e.g., interrupted or thresholds
                not met).

        Notes:
            - This call **blocks** until completion or failure/stop.
            - If both joint-space and task-space thresholds are given, **all** given
            thresholds must be satisfied.
        """
        return super().trapj_mwait(
            goal_angles,
            goal_angvels,
            peak_angvels,
            peak_angaccs,
            duration,
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
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

    def attrj_mwait(
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
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=10000),
        angles_enorm_thold: Optional[float] = 2,
        pos_enorm_thold: Optional[float] = None,
        rot_enorm_thold: Optional[float] = None,
        angvels_enorm_thold: Optional[float] = 4,
        vel_enorm_thold: Optional[float] = None,
        w_enorm_thold: Optional[float] = None,
    ) -> bool:
        """
        Generate and execute an attractor-based joint-space (attrj) trajectory,
        then block until the goal is reached or the timeout is exceeded.

        The motion is generated using an attractor control law with proportional
        gains (`kp`) in joint space, while respecting per-joint velocity and
        acceleration limits.

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

            timeout (datetime.timedelta, default=10s):
                Maximum time to wait for motion completion.

            angles_enorm_thold (float, optional, default=2):
                Threshold for joint-angle error norm [deg].

            pos_enorm_thold (float, optional):
                Threshold for end-effector position error norm [m].

            rot_enorm_thold (float, optional):
                Threshold for end-effector rotation error norm [deg].

            angvels_enorm_thold (float, optional, default=4):
                Threshold for joint-velocity error norm [deg/s].

            vel_enorm_thold (float, optional):
                Threshold for end-effector linear-velocity error norm [m/s].

            w_enorm_thold (float, optional):
                Threshold for end-effector angular-velocity error norm [deg/s].

        Returns:
            bool:
                True if the motion completes within the thresholds before the timeout;
                False otherwise.

        Notes:
            - This call **blocks** until completion or timeout.
            - If both joint-space and task-space thresholds are provided, **all**
            must be satisfied for success.
            - Use `attrj` if you want to start motion without waiting.
        """
        return super().attrj_mwait(
            goal_angles,
            kp,
            goal_angvels,
            peak_angvels,
            peak_angaccs,
            timeout,
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
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

    def attrl_mwait(
        self,
        goal_tmat: npt.NDArray[np.generic],
        kp: float = 50,
        goal_a: npt.NDArray[np.generic] = np.zeros(6),
        peak_endvel: float = 0.5,
        peak_endangvel: float = 180,
        peak_endacc: float = sys.float_info.max,
        peak_endangacc: float = sys.float_info.max,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=10000),
        angles_enorm_thold: Optional[float] = None,
        pos_enorm_thold: Optional[float] = 0.002,
        rot_enorm_thold: Optional[float] = 3,
        angvels_enorm_thold: Optional[float] = None,
        vel_enorm_thold: Optional[float] = 0.004,
        w_enorm_thold: Optional[float] = 6,
    ) -> bool:
        """
        Generate and execute an attractor-based Cartesian-space (attrl) trajectory,
        then block until the goal is reached or the timeout is exceeded.

        The motion is generated using an attractor control law with proportional
        gain (`kp`) in task space (Cartesian), while respecting linear/angular
        velocity and acceleration limits.

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

            timeout (datetime.timedelta, default=10s):
                Maximum time to wait for motion completion.

            angles_enorm_thold (float, optional):
                Threshold for joint-angle error norm [deg].

            pos_enorm_thold (float, optional, default=0.002):
                Threshold for end-effector position error norm [m].

            rot_enorm_thold (float, optional, default=3):
                Threshold for end-effector rotation error norm [deg].

            angvels_enorm_thold (float, optional):
                Threshold for joint-velocity error norm [deg/s].

            vel_enorm_thold (float, optional, default=0.004):
                Threshold for end-effector linear-velocity error norm [m/s].

            w_enorm_thold (float, optional, default=6):
                Threshold for end-effector angular-velocity error norm [deg/s].

        Returns:
            bool:
                True if the motion completes within the thresholds before the timeout;
                False otherwise.

        Notes:
            - This call **blocks** until completion or timeout.
            - If both joint-space and task-space thresholds are provided, **all**
            must be satisfied for success.
            - Use `attrl` if you want to start motion without waiting.
        """
        return super().attrl_mwait(
            goal_tmat,
            kp,
            goal_a,
            peak_endvel,
            peak_endangvel,
            peak_endacc,
            peak_endangacc,
            timeout,
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
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

        This method plays a sequence of joint-space goal configurations, optionally
        with per-waypoint target velocities and accelerations, while respecting
        per-joint peak velocity and acceleration limits. The motion is started and
        this call returns immediately without waiting for completion.

        Args:
            goal_angles_set (ndarray, shape: (N, 6), dtype=np.generic):
                Sequence of target joint angles [deg] for each waypoint (N waypoints).

            goal_angvels_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint velocities [deg/s] per waypoint.
                If None, zeros are assumed.

            goal_angaccs_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint accelerations [deg/s²] per waypoint.
                If None, zeros are assumed.

            peak_angvels (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum velocities [deg/s]. Must be strictly positive.

            peak_angaccs (ndarray, shape: (6,), dtype=np.generic, default=[1200,1200,1800,2250,2250,2550]):
                Per-joint maximum accelerations [deg/s²]. Must be strictly positive.

        Returns:
            bool:
                True if the trajectory was successfully generated and started;
                False if inputs are invalid or the system is in an invalid state.

        Notes:
            - The first dimension N (number of waypoints) must be the same for
            `goal_angles_set`, and for `goal_angvels_set`/`goal_angaccs_set` when provided.
            - This call **does not** wait for motion completion.
            - Use `playj_mwait` if you need to block until all waypoints are reached.
        """
        return super().playj(
            goal_angles_set,
            goal_angvels_set,
            goal_angaccs_set,
            peak_angvels,
            peak_angaccs,
        )

    def playj_mwait(
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
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=10000),
    ) -> bool:
        """
        Generate and execute a joint-space playback (playj) trajectory,
        then block until all waypoints are reached or the timeout is exceeded.

        This method plays a sequence of joint-space goal configurations, optionally
        with per-waypoint target velocities and accelerations, while respecting
        per-joint peak velocity and acceleration limits. The call blocks until the
        final waypoint is reached or the timeout elapses.

        Args:
            goal_angles_set (ndarray, shape: (N, 6), dtype=np.generic):
                Sequence of target joint angles [deg] for each waypoint (N waypoints).

            goal_angvels_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint velocities [deg/s] per waypoint.
                If None, zeros are assumed.

            goal_angaccs_set (ndarray, shape: (N, 6), dtype=np.generic, optional):
                Sequence of target joint accelerations [deg/s²] per waypoint.
                If None, zeros are assumed.

            peak_angvels (ndarray, shape: (6,), dtype=np.generic, default=[120,120,180,225,225,255]):
                Per-joint maximum velocities [deg/s]. Must be strictly positive.

            peak_angaccs (ndarray, shape: (6,), dtype=np.generic, default=[1200,1200,1800,2250,2250,2550]):
                Per-joint maximum accelerations [deg/s²]. Must be strictly positive.

            timeout (datetime.timedelta, default=10s):
                Maximum time to wait for all waypoints to be reached.

        Returns:
            bool:
                True if the motion completes within the timeout; False otherwise
                (e.g., interrupted, limits violated, or timeout).

        Notes:
            - This call **blocks** until all waypoints are reached or the timeout occurs.
            - The first dimension N (number of waypoints) must be the same for
            `goal_angles_set`, and for `goal_angvels_set`/`goal_angaccs_set` when provided.
            - Use `playj` if you want to start the motion without waiting.
        """
        return super().playj_mwait(
            goal_angles_set,
            goal_angvels_set,
            goal_angaccs_set,
            peak_angvels,
            peak_angaccs,
            timeout,
        )

    def mwait(
        self,
        timeout: datetime.timedelta = datetime.timedelta(milliseconds=10000),
        angles_enorm_thold: Optional[float] = 2,
        pos_enorm_thold: Optional[float] = 0.002,
        rot_enorm_thold: Optional[float] = 3,
        angvels_enorm_thold: Optional[float] = 4,
        vel_enorm_thold: Optional[float] = 0.004,
        w_enorm_thold: Optional[float] = 6,
    ) -> bool:
        """
        Block until the current motion completes or the timeout is exceeded.

        This method monitors the active trajectory and blocks until all provided
        thresholds are satisfied or the timeout duration has elapsed.

        Args:
            timeout (datetime.timedelta, default=10s):
                Maximum time to wait for motion completion.

            angles_enorm_thold (float, optional, default=2):
                Threshold for joint-angle error norm [deg].

            pos_enorm_thold (float, optional, default=0.002):
                Threshold for end-effector position error norm [m].

            rot_enorm_thold (float, optional, default=3):
                Threshold for end-effector rotation error norm [deg].

            angvels_enorm_thold (float, optional, default=4):
                Threshold for joint-velocity error norm [deg/s].

            vel_enorm_thold (float, optional, default=0.004):
                Threshold for end-effector linear-velocity error norm [m/s].

            w_enorm_thold (float, optional, default=6):
                Threshold for end-effector angular-velocity error norm [deg/s].

        Returns:
            bool:
                True if the motion completes within the thresholds before the timeout;
                False otherwise.

        Notes:
            - This call **blocks** until completion or timeout.
            - If both joint-space and task-space thresholds are provided, **all**
            must be satisfied for success.
        """
        return super().mwait(
            timeout,
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
        )

    ###################################################

    def get_robot_state(self) -> Optional[RobotState]:
        data = super().get_robot_state()

        if data is None:
            return None

        return RobotState(data)

    def get_current_angles(self) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array with deg unit
        return super().get_current_angles()

    def get_current_angvels(self) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array with deg/s unit
        return super().get_current_angvels()

    def get_current_tmat(self) -> Optional[npt.NDArray[np.float64]]:
        # return (4, 4) numpy array
        return super().get_current_tmat()

    def get_current_jmat(self) -> Optional[npt.NDArray[np.float64]]:
        # return (6, 6) numpy array
        return super().get_current_jmat()

    def get_current_a(self) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array
        return super().get_current_a()

    def get_current_joint_torque(self) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array with Nm unit
        return super().get_current_joint_torque()

    def get_current_joint_external_torque(
        self,
    ) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array with Nm unit
        return super().get_current_joint_external_torque()

    def get_current_end_external_force(
        self,
    ) -> Optional[npt.NDArray[np.float64]]:
        # return (6,) numpy array with N, Nm unit
        # (force_x, force_y, force_z, torque_x, torque_y, torque_z)
        return super().get_current_end_external_force()

    ###################################################

    def get_traj_state(self) -> TrajState:
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
        return TrajState(super().get_traj_state())

    def get_desired_angles(self) -> npt.NDArray[np.float64]:
        """
        Get the desired joint angles for the current trajectory.

        This method returns the target joint angles that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (6, ) desired joint angles [deg].
        """
        return super().get_desired_angles()

    def get_desired_angvels(self) -> npt.NDArray[np.float64]:
        """
        Get the desired joint angular velocities for the current trajectory.

        This method returns the target joint angular velocities that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (6, ) desired joint angular velocities [deg/s].
        """
        return super().get_desired_angvels()

    def get_desired_tmat(self) -> npt.NDArray[np.float64]:
        """
        Get the desired end-effector transformation matrix for the current trajectory.

        This method returns the target end-effector transformation matrix that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (4, 4) desired end-effector transformation matrix [m].
        """
        return super().get_desired_tmat()

    def get_desired_jmat(self) -> npt.NDArray[np.float64]:
        """
        Get the desired end-effector Jacobian matrix for the current trajectory.

        This method returns the target end-effector Jacobian matrix that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (6, 6) desired end-effector Jacobian matrix.
        """
        return super().get_desired_jmat()

    def get_desired_a(self) -> npt.NDArray[np.float64]:
        """
        Get the desired end-effector screw velocity for the current trajectory.

        This method returns the target end-effector screw velocity that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (6, ) desired end-effector screw velocity [m/s, rad/s].
        """
        return super().get_desired_a()

    def get_desired_a_d1(self) -> npt.NDArray[np.float64]:
        """
        Get the desired end-effector screw acceleration for the current trajectory.

        This method returns the target end-effector screw acceleration that the robot is currently
        trying to achieve in the ongoing trajectory.

        Returns:
            (6, ) desired end-effector screw acceleration [m/s², rad/s²].
        """
        return super().get_desired_a_d1()

    ####################################################

    def get_goal_angles(self) -> Optional[npt.NDArray[np.float64]]:
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
        return super().get_goal_angles()

    def get_goal_angvels(self) -> Optional[npt.NDArray[np.float64]]:
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
        return super().get_goal_angvels()

    def get_goal_tmat(self) -> Optional[npt.NDArray[np.float64]]:
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
        return super().get_goal_tmat()

    def get_goal_a(self) -> Optional[npt.NDArray[np.float64]]:
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
        return super().get_goal_a()

    def get_goal_reached(
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
        return super().get_goal_reached(
            angles_enorm_thold,
            pos_enorm_thold,
            rot_enorm_thold,
            angvels_enorm_thold,
            vel_enorm_thold,
            w_enorm_thold,
        )

    ####################################################

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
