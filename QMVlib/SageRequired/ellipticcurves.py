from manimlib import *
from sage.all import *


class EllipticCurveMob(VMobject):
    def __init__(self, *a_invars, E=None):
        if E is None and len(a_invars) != 2 and len(a_invars) != 5:
            raise ValueError('EllipticCurveMob takes an elliptic curve or a_invars')
        
        if E is not None:
            a1, a2, a3, a4, a6 = E.ainvs()
        elif len(a_invars) == 2:
            a1, a2, a3, a4, a6 = 0, 0, 0, a_invars[0], a_invars[1]
            E = EllipticCurve([a1, a2, a3, a4, a6])
        else:
            a1, a2, a3, a4, a6 = a_invars
            E = EllipticCurve([a1, a2, a3, a4, a6])

        RR = RealField()
        d = E.division_polynomial(2)
        r = sorted(d.roots(RR, multiplicities=False))

        def f1(z):
            # Internal function for plotting first branch of the curve
            return (-(a1 * z + a3) + sqrt(abs(d(z)))) / 2

        def f2(z):
            # Internal function for plotting second branch of the curve
            return (-(a1 * z + a3) - sqrt(abs(d(z)))) / 2
        xmins = []
        xmaxs = []
        if len(r) > 1:
            xmins.append(r[0])
            xmaxs.append(r[1])

        # The following 3 is an aesthetic choice.  It's possible
        # that we should compute both of the following when
        # components=='both' and len(r) > 1 and take the maximum
        # generated xmax.
        if len(r) == 1 or r[2] - r[1] > 3 * (r[1] - r[0]):
            flex = sorted(E.division_polynomial(3).roots(RR, multiplicities=False))
            flex = flex[-1]
            xmins.append(r[-1])
            # The doubling here is an aesthetic choice
            xmaxs.append(flex + 2 * (flex - r[-1]))
        else:
            # First the easy part.
            xmins.append(r[-1])
            # There are two components and the unbounded component
            # is not too far from the bounded one.  We scale so
            # that the unbounded component is twice as tall as the
            # bounded component.  The y values corresponding to
            # horizontal tangent lines are determined as follows.
            # We implicitly differentiate the equation for this
            # curve and get
            # 2 yy' + a1 y + a1 xy' + a3 y' = 3 x^2 + 2a2 x + a4

            R = RR['x']
            x = R.gen()
            if a1 == 0:
                # a horizontal tangent line can only occur at a root of
                Ederiv = 3 * x ** 2 + 2 * a2 * x + a4
            else:
                # y' = 0  ==>  y = (3*x^2 + 2*a2*x + a4) / a1
                y = (3 * x ** 2 + 2 * a2 * x + a4) / a1
                Ederiv = y ** 2 + a1 * x * y + a3 * y - (x ** 3 + a2 * x ** 2 + a4 * x + a6)
            critx = [a for a in Ederiv.roots(RR, multiplicities=False)
                     if r[0] < a < r[1]]
            if not critx:
                raise RuntimeError("No horizontal tangent lines on bounded component")
            # The 2.5 here is an aesthetic choice
            ymax = 2.5 * max([f1(a) for a in critx])
            ymin = 2.5 * min([f2(a) for a in critx])
            top_branch = ymax ** 2 + a1 * x * ymax + a3 * ymax - (x ** 3 + a2 * x ** 2 + a4 * x + a6)
            bottom_branch = ymin ** 2 + a1 * x * ymin + a3 * ymin - (x ** 3 + a2 * x ** 2 + a4 * x + a6)
            xmaxs.append(
                max(top_branch.roots(RR, multiplicities=False) + bottom_branch.roots(RR, multiplicities=False)))
        xmins = min(xmins)
        xmaxs = max(xmaxs)
        span = xmaxs - xmins
        xmin = xmins - .02 * span
        xmax = xmaxs + .02 * span
        number_plane = NumberPlane(
            x_range=[xmin, xmax],
            y_range=[-f1(xmax), f1(xmax)],
            width=4,
            height=2,
            background_line_style={
                "stroke_width": 1.2,
                "stroke_opacity": 0.2
            }
        ).shift(UP * 0.5)
        EC_plane = [number_plane]
        # EC_plane = []

        I = []
        if xmax > r[-1]:
            # one real root; 1 component
            if xmin <= r[-1]:
                I.append((r[-1], xmax, '<'))
            else:
                I.append((xmin, xmax, '='))
        if len(r) > 1 and (xmin < r[1] or xmax > r[0]):
            if xmin <= r[0]:
                if xmax >= r[1]:
                    I.append((r[0], r[1], 'o'))
                else:
                    I.append((r[0], xmax, '<'))
            elif xmax >= r[1]:
                I.append((xmin, r[1], '>'))
            else:
                I.append((xmin, xmax, '='))


        for j in range(len(I)):
            a, b, shape = I[j]
            component = [number_plane.plot(f1, x_range=[a, b, (b - a)/100], color=YELLOW_C), number_plane.plot(f2, x_range=[a, b, (b - a)/100], color=YELLOW_C)]
            if shape == 'o':
                component[0].shift(LEFT * 0.1)
                component[1].shift(LEFT * 0.1)
            EC_plane.append(component[0])
            EC_plane.append(component[1])

        return VGroup(*EC_plane).shift(OUT)