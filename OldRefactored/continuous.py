import numpy as np
from manimlib import *
import time
from datetime import datetime


class SpaceFillingCurveScene(InteractiveScene):
    def construct(self):
        #Generate curve mobjects
        order = 8
        hilbertCurves = []
        hilbert_label = []
        for i in range(order):
            timestamp = datetime.fromtimestamp(time.time())
            print('Generating curve ', i, ' TIMESTAMP: ', timestamp)
            curve = self.hilbertCurve(order=i + 1)
            curve.center().shift(UP * 0.5)
            hilbertCurves.append(curve)
            hilbert_label.append(Tex(R'H_' + str(i) + R': [0, 1] \to [0, 1]^2')
                                 .to_edge(DOWN).shift(RIGHT).scale(0.7))

        number_line = NumberLine(x_range=[0, 1], include_numbers=True)
        number_line.next_to(hilbert_label[0], LEFT, buff=1)
        tracker = ValueTracker(0)
        dot = always_redraw(lambda: Dot(number_line.n2p(tracker.get_value()), color=BLUE_E))

        transformations = []
        for i in range(order - 1):
            timestamp = datetime.fromtimestamp(time.time())
            print('Generating transform animation ', i, '->', (i + 1), ' TIMESTAMP: ', timestamp)
            transformations.append(ReplacementTransform(hilbertCurves[i], hilbertCurves[i + 1], path_arc=TAU * 1/3))

        # Draw 4 curves, showing tracker value
        self.play(ShowCreation(hilbertCurves[0]), Write(hilbert_label[0], run_time=0.4), FadeIn(number_line, run_time=0.4))
        for i in range(4):
            self.play(FadeOut(hilbertCurves[i]), run_time=0.5)
            tracker.set_value(0)
            self.add(dot)
            dot_animation = AnimationGroup(tracker.animate.set_value(1), rate_func=linear, run_time=0.5 * (i + 1))
            self.play(TransformMatchingTex(hilbert_label[i], hilbert_label[i + 1], run_time=0.5),
                      ShowCreation(hilbertCurves[i + 1], rate_func=linear, run_time=0.5 * (i + 1)),
                      dot_animation
                      )
            self.play(FadeOut(dot), run_time=0.5)

        # Do more transformations
        self.play(self.frame.animate.scale(0.4))
        for i in range(4, order - 1):
            self.play(transformations[i])
            self.wait(1)

        self.play(self.frame.animate.scale(1/0.4))
        self.wait(1)

    def hilbertCurve(self, order=1, x_initial=0, y_initial=0,
                     colors=[PURPLE_B, MAROON_B, BLUE_C, TEAL_C, YELLOW_C, GREEN_C, LIGHT_BROWN]):
        segment_length = 5 / (2 ** order)
        stroke_width = (-3 / 7) * order + (31 / 7)

        directions = self.hilbert(level=order)
        prevCenter = np.array((x_initial, y_initial, 0.))
        segments = [prevCenter]

        for direct in directions:
            dirTranslate = UP
            if direct == 'LEFT':
                dirTranslate = LEFT
            if direct == 'RIGHT':
                dirTranslate = RIGHT
            if direct == 'DOWN':
                dirTranslate = DOWN

            endPt = np.copy(prevCenter + (segment_length * dirTranslate))
            segments.append(endPt)
            prevCenter = endPt

        segments = VMobject(stroke_width=stroke_width).set_points_as_corners(segments)
        segments.set_color_by_gradient(*colors)
        return segments

    def hilbert(self, level=1, direction='UP'):
        directions = []
        if level == 1:
            if direction == 'LEFT':
                directions.append('RIGHT')
                directions.append('DOWN')
                directions.append('LEFT')
            if direction == 'RIGHT':
                directions.append('LEFT')
                directions.append('UP')
                directions.append('RIGHT')
            if direction == 'UP':
                directions.append('DOWN')
                directions.append('RIGHT')
                directions.append('UP')
            if direction == 'DOWN':
                directions.append('UP')
                directions.append('LEFT')
                directions.append('DOWN')
        else:
            if direction == 'LEFT':
                directions.extend(self.hilbert(level - 1, 'UP'))
                directions.append('RIGHT')
                directions.extend(self.hilbert(level - 1, 'LEFT'))
                directions.append('DOWN')
                directions.extend(self.hilbert(level - 1, 'LEFT'))
                directions.append('LEFT')
                directions.extend(self.hilbert(level - 1, 'DOWN'))
            if direction == 'RIGHT':
                directions.extend(self.hilbert(level - 1, 'DOWN'))
                directions.append('LEFT')
                directions.extend(self.hilbert(level - 1, 'RIGHT'))
                directions.append('UP')
                directions.extend(self.hilbert(level - 1, 'RIGHT'))
                directions.append('RIGHT')
                directions.extend(self.hilbert(level - 1, 'UP'))
            if direction == 'UP':
                directions.extend(self.hilbert(level - 1, 'LEFT'))
                directions.append('DOWN')
                directions.extend(self.hilbert(level - 1, 'UP'))
                directions.append('RIGHT')
                directions.extend(self.hilbert(level - 1, 'UP'))
                directions.append('UP')
                directions.extend(self.hilbert(level - 1, 'RIGHT'))
            if direction == 'DOWN':
                directions.extend(self.hilbert(level - 1, 'RIGHT'))
                directions.append('UP')
                directions.extend(self.hilbert(level - 1, 'DOWN'))
                directions.append('LEFT')
                directions.extend(self.hilbert(level - 1, 'DOWN'))
                directions.append('DOWN')
                directions.extend(self.hilbert(level - 1, 'LEFT'))
        return directions
