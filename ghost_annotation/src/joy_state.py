from dataclasses import dataclass, field
from typing import Optional

from sensor_msgs.msg import Joy as JoyMsg


class XboxController:
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LEFT_BUMPER = 4
    BUTTON_RIGHT_BUMPER = 5
    BUTTON_LEFT_CENTER = 6
    BUTTON_RIGHT_CENTER = 7
    BUTTON_CENTER = 8
    BUTTON_LEFT_STICK = 9
    BUTTON_RIGHT_STICK = 10

    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_LEFT_TRIGGER = 2
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_RIGHT_TRIGGER = 5
    AXIS_DPAD_HORIZONTAL = 6
    AXIS_DPAD_VERTICAL = 7

    DPAD_UP_VAL = 1
    DPAD_DOWN_VAL = -1
    DPAD_LEFT_VAL = 1
    DPAD_RIGHT_VAL = -1


@dataclass
class JoyState:
    prev_axes: list[float] = field(default_factory=list)
    axes: list[float] = field(default_factory=list)

    prev_buttons: list[int] = field(default_factory=list)
    buttons: list[int] = field(default_factory=list)
    button_held_refs: list[float] = field(default_factory=list)

    dt: float = 0.0
    stamp: float = 0.0
    continuous: bool = False

    def update(self, joy: JoyMsg) -> None:
        self.continuous = (
            len(joy.axes) == len(self.axes) and
            len(joy.buttons) == len(self.buttons)
        )

        self.prev_axes = self.axes
        self.prev_buttons = self.buttons

        self.axes = list(joy.axes)
        self.buttons = list(joy.buttons)

        t = (joy.header.stamp.sec + joy.header.stamp.nanosec * 1e-9)

        if self.stamp > 0.0:
            self.dt = t - self.stamp

        self.stamp = t

        for i, b in enumerate(self.buttons):
            if i < len(self.button_held_refs):
                if not bool(b):
                    self.button_held_refs[i] = t
            else:
                self.button_held_refs.append(t)

    def update_disconnected(self) -> None:
        self.continuous = False
        self.axes.clear()
        self.buttons.clear()
        self.dt = 0.0
        self.stamp = 0.0

    # ------------------------------------------------------------------
    # Index checks
    # ------------------------------------------------------------------

    def has_button_idx(self, idx: int) -> bool:
        return 0 <= idx < len(self.buttons)

    def has_axis_idx(self, idx: int) -> bool:
        return 0 <= idx < len(self.axes)

    def is_button_continuous(self, idx: int) -> bool:
        return self.continuous and self.has_button_idx(idx)

    def is_axis_continuous(self, idx: int) -> bool:
        return self.continuous and self.has_axis_idx(idx)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def get_raw_button(self, idx: int) -> bool:
        return self.has_button_idx(idx) and bool(self.buttons[idx])

    def get_button_pressed(self, idx: int) -> bool:
        return (
            self.is_button_continuous(idx)
            and not self.prev_buttons[idx]
            and self.buttons[idx]
        )

    def get_button_released(self, idx: int) -> bool:
        return (
            self.is_button_continuous(idx)
            and self.prev_buttons[idx]
            and not self.buttons[idx]
        )

    def get_button_held(self, idx: int, thresh: float) -> bool:
        return (
            self.has_button_idx(idx) and
            (self.stamp - self.button_held_refs[idx]) >= thresh
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    def get_raw_axis(self, idx: int) -> float:
        return self.axes[idx] if self.has_axis_idx(idx) else 0.0

    def get_trigger_axis(self, idx: int) -> float:
        return (1.0 - self.axes[idx]) / 2.0 if self.has_axis_idx(idx) else 0.0

    def get_axis_delta(self, idx: int) -> float:
        if self.is_axis_continuous(idx):
            return self.axes[idx] - self.prev_axes[idx]
        return 0.0

    def get_axis_velocity(self, idx: int) -> float:
        if self.is_axis_continuous(idx) and (self.dt > 0.0):
            return (self.axes[idx] - self.prev_axes[idx]) / self.dt
        return 0.0

    def get_trapezoid_sum(self, idx: int) -> float:
        if self.is_axis_continuous(idx):
            return (self.axes[idx] + self.prev_axes[idx]) * (0.5 * self.dt)
        return 0.0

    def get_trigger_trapezoid_sum(self, idx: int) -> float:
        if self.is_axis_continuous(idx):
            return (
                (2.0 - self.prev_axes[idx] - self.axes[idx])
                * 0.25
                * self.dt
            )
        return 0.0

    # ------------------------------------------------------------------
    # POV (D-pad axes)
    # ------------------------------------------------------------------

    def get_raw_pov(self, idx: int, sgn: int) -> bool:
        return self.has_axis_idx(idx) and (int(self.axes[idx]) * sgn) > 0

    def get_pov_pressed(self, idx: int, sgn: int) -> bool:
        return (
            self.is_axis_continuous(idx)
            and (int(self.prev_axes[idx]) * sgn) <= 0
            and (int(self.axes[idx]) * sgn) > 0
        )

    def get_pov_released(self, idx: int, sgn: int) -> bool:
        return (
            self.is_axis_continuous(idx)
            and (int(self.prev_axes[idx]) * sgn) > 0
            and (int(self.axes[idx]) * sgn) <= 0
        )



@dataclass
class JoyButton:
    joy: JoyState
    idx: int = 0

    def _joy(self, joy: Optional[JoyState]) -> JoyState:
        return self.joy if joy is None else joy

    def raw_value(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_raw_button(self.idx)

    def was_pressed(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_button_pressed(self.idx)

    def was_released(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_button_released(self.idx)

    def was_held(self, thresh: float, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_button_held(self.idx, thresh)


@dataclass
class JoyAxis:
    joy: JoyState
    idx: int = 0

    def _joy(self, joy: Optional[JoyState]) -> JoyState:
        return self.joy if joy is None else joy

    def raw_value(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_raw_axis(self.idx)

    def deadzone_value(
        self, deadzone: float, joy: Optional[JoyState] = None
    ) -> float:
        v = self.raw_value(joy)
        return v if abs(v) >= deadzone else 0.0

    def trigger_value(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_trigger_axis(self.idx)

    def delta(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_axis_delta(self.idx)

    def velocity(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_axis_velocity(self.idx)

    def trapezoid_sum(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_trapezoid_sum(self.idx)

    def trigger_trapezoid_sum(self, joy: Optional[JoyState] = None) -> float:
        return self._joy(joy).get_trigger_trapezoid_sum(self.idx)


@dataclass
class JoyPov:
    joy: JoyState
    idx: int = 0
    sgn: int = 0

    def _joy(self, joy: Optional[JoyState]) -> JoyState:
        return self.joy if joy is None else joy

    def raw_value(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_raw_pov(self.idx, self.sgn)

    def was_pressed(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_pov_pressed(self.idx, self.sgn)

    def was_released(self, joy: Optional[JoyState] = None) -> bool:
        return self._joy(joy).get_pov_released(self.idx, self.sgn)
