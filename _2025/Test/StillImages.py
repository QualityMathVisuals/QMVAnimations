from QMVlib import *
from sage.all import *
from manimlib import *


class StillImageScene(InteractiveScene):
    def construct(self):
        # Draw some images
        qmv_fancy = VGroup(
            Tex(
                R'Q',
                font_size=144
            ), 
            Tex(
                R'M',
                font_size=122
            ), 
            Tex(
                R'V',
                font_size=122
            ), 
        ).arrange(RIGHT, buff=0)

        qmv_long = VGroup(
            Tex(
                R'Q',
                font_size=104
            ), 
            Text('uality', font_size=96),
            Tex(
                R'M',
                font_size=104
            ), 
            Text('ath ', font_size=96),
            Tex(
                R'V',
                font_size=104
            ), 
            Text('isuals ', font_size=96),
        ).arrange(RIGHT, buff=0.05)
        qmv_long[0].shift(LEFT * 0.05)
        qmv_long[1].shift(LEFT * 0.05)
        qmv_long[2].shift(RIGHT * 0.1)
        qmv_long[4].shift(RIGHT * 0.1)
        qmv_long[5].shift(RIGHT * 0.05)
        self.add(qmv_long)

        qmv_long[0].shift(LEFT * 0.1)
        blus = [QMV_BLUE_D, QMV_BLUE_C, QMV_BLUE_B]
        grads = color_gradient([QMV_BLUE_C, QMV_PURPLE_A, QMV_BLUE_B, QMV_PINK_A], 100)
        qmv_fancy[0]
        qmv_fancy[1]
        qmv_fancy[2].shift(RIGHT * 0.08 + DOWN *0.02)
        qmv_fancy.scale(2)
        self.add(qmv_fancy)

        #Add elliptic Curve
        A, B = -4, 9
        E = EllipticCurve([A, B])
        RR = RealField()
        a1, a2, a3, a4, a6 = 0, 0, 0, A, B
        RR = RealField()
        d = E.division_polynomial(2)
        r = sorted(d.roots(RR, multiplicities=False))
        x_min = r[0]
        x_max = 5
        xs = np.linspace(x_min, x_max, 1000)

        def f1(z):
            # Internal function for plotting first branch of the curve
            return (-(a1 * z + a3) + sqrt(abs(z**3 + A * z + B))) / 2.2

        def f2(z):
            # Internal function for plotting second branch of the curve
            return (-(a1 * z + a3) - sqrt(abs(z**3 + A * z + B))) / 2.2
        
        xs = xs[::-1]
        y1 = f1(xs)
        xs = xs[::-1]
        y2 = f2(xs)
        xs = xs[::-1]
        points_y1 = np.array([[x, y, 0] for x, y in zip(xs, y1)])
        xs = xs[::-1]
        points_y2 = np.array([[x, y, 0] for x, y in zip(xs, y2)])
        all_points = np.vstack((points_y1, points_y2))
        ec = VMobject().set_stroke(width=10).set_points_smoothly(all_points)
        ec.scale(1)
        self.add(ec)
        F2 = E.division_field(2)
        E = EllipticCurve([F2(A), F2(B)])
        P1 = E(0, 3, 1)
        relative_points = [
            [0, f1(0), 0],
            [3, f1(3), 0],
            [-2.6, f1(-2.6), 0],
            [-2.6, f2(-2.6), 0]
        ]
        points_manim = [
            Dot(pt, fill_color=TEAL)
            for pt in relative_points
        ]

        line1 = Line(start=relative_points[1], end=relative_points[2]).set_color(GREY_B)
        line2 = Line(start=relative_points[2], end=relative_points[3]).set_color(GREY_B)
        line1.z_index = -2
        line2.z_index = -2
        ec.z_index = -3
        self.add(*points_manim, line1, line2)
        VGroup(*points_manim, ec, line1, line2).flip(UP)
        ec.set_color_by_gradient(*[QMV_BLUE_C, QMV_PURPLE_C, QMV_PINK_C, QMV_PURPLE_B, QMV_BLUE_D, QMV_BLUE_C, QMV_BLUE_A, QMV_PURPLE_A, QMV_PINK_B, QMV_PINK_D, QMV_PURPLE_B, QMV_PURPLE_D, QMV_BLUE_D])
        # qmv_fancy.shift(LEFT * 1.5)
        self.remove(qmv_long)
        VGroup(*points_manim, ec, line1, line2, qmv_fancy)