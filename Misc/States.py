from enum import Enum

class State(Enum):
    IDLE = "IDLE"
    BACKSWING = "BACKSWING"
    FORWARD_SWING = "FORWARD_SWING"
    TRACKING_BALL = "TRACKING_BALL"