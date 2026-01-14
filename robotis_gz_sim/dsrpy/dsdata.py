"""data.py"""

import datetime
from typing import Union

import robotis_gz_sim.dsrpy.dsrbind as _dsrb
from . import dsenum as dn

__all__ = [
    "LogAlarm",
]


class LogAlarm:
    """LogAlarm"""

    time: datetime.datetime
    level: dn.LogLevel
    group: dn.LogGroup
    index: Union[
        int,
        dn.LogGroupSystemFMK,
        dn.LogGroupMotionLib,
        dn.LogGroupSafetyController,
    ]
    param1: str
    param2: str
    param3: str

    def __init__(self, alarm: _dsrb.LogAlarm) -> None:
        self.time = alarm.time

        self.level = dn.LogLevel(alarm.iLevel)
        self.group = dn.LogGroup(alarm.iGroup)

        try:
            if self.group == dn.LogGroup.LOG_GROUP_SYSTEMFMK:
                self.index = dn.LogGroupSystemFMK(alarm.iIndex)
            elif self.group == dn.LogGroup.LOG_GROUP_MOTIONLIB:
                self.index = dn.LogGroupMotionLib(alarm.iIndex)
            elif self.group == dn.LogGroup.LOG_GROUP_SAFETYCONTROLLER:
                self.index = dn.LogGroupSafetyController(alarm.iIndex)
            else:
                self.index = alarm.iIndex

        except ValueError:
            self.index = alarm.iIndex

        self.param1 = alarm.szParam1
        self.param2 = alarm.szParam2
        self.param3 = alarm.szParam3

    def __str__(self) -> str:
        return f"LogAlarm({self.time}, {self.level}, {self.group}, {self.index}, {self.param1}, {self.param2}, {self.param3})"
