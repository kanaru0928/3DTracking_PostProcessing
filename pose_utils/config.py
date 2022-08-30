from enum import Enum

class InterpolationType(Enum):
    NONE = 0
    LINER = 1
    ACCELERATION = 2
    RESISTANCE = 3

class Config:
    interpolationType: InterpolationType = InterpolationType.ACCELERATION
    isSwapAvailable: bool = True
    detectNet: str = 'yolov5s'
    isPowerReleaseMode = False
