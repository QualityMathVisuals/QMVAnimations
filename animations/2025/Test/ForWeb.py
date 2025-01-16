from manimlib import *
import sys
sys.path.insert(0, '/Users/nswanson/QMVManimGL/animations/')
from QMVlib import * 


class Pics(InteractiveScene):
    def construct(self):
        self.wait(0)
        self.add(Square(color=MY_PURPLE_A))
        self.add(Circle(color=MY_PURPLE_A).shift(RIGHT))
