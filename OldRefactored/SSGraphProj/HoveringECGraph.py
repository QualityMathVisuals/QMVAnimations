from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup
from manim.renderer.opengl_renderer import OpenGLCamera
from sage.all import *
from SSGraph import TraverseSSGraphScene
from SupersingularGraphs.SSGraph import BreadthFirstSSGraphScene


class HoveringECGraphScene(BreadthFirstSSGraphScene, ThreeDScene):
    def __init__(self, p= (2**216)*(3**137)-1, l=3):
        super().__init__(p , l)
        self.fade_factor = 1
        self.ring_spacing_range = [self.vertex_radius * 5, self.vertex_radius * (6  + self.l)]
        self.ring_spacing_min_scaling = 0.6

    def setup(self):
        super().setup()
        self.begin_ambient_camera_rotation(0.2)

    def j_vertex(self, j_inv, vertex_color=BLUE_B):
        vert = super().j_vertex(j_inv, vertex_color=vertex_color)
        return OpenGLVGroup(vert[0], vert[1], self.elliptic_curve_mob(j_inv))

    def elliptic_curve_mob(self, j_inv):
        RR = RealField()
        if j_inv == 1728:
            E = EllipticCurve([1, 0])
        elif j_inv.to_integer() > 99999:
            j_inv = randrange(1, 99999)
            E = EllipticCurve_from_j(Integer(j_inv))
        else:
            E = EllipticCurve_from_j(j_inv.to_integer())

        a1, a2, a3, a4, a6 = E.ainvs()
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
        number_plane = Axes(
            x_range=[xmin, xmax],
            y_range=[-f1(xmax), f1(xmax)],
            x_length=2,
            y_length=2,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": False},
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

        return OpenGLVGroup(*EC_plane).shift(OUT)

    def construct(self):
        initial_dot = self.vertices[self.j_invars[0]]
        self.add(initial_dot)
        self.move_camera(phi=45*DEGREES, run_time=2)
        self.wait(3)
        self.draw_next_level(animate_each_node=True)
        self.wait(3)
        scaling_factor = 1.5
        self.play(self.camera.animate.set_height(self.camera.get_height() * scaling_factor),
        self.camera.animate.set_width(self.camera.get_width() * scaling_factor))
        self.draw_next_level(animate_each_node=True)
        self.wait(3)
        self.play(self.camera.animate.set_height(self.camera.get_height() * scaling_factor),
                  self.camera.animate.set_width(self.camera.get_width() * scaling_factor))
        self.draw_next_level(animate_each_node=True)
        self.wait(3)
        self.play(self.camera.animate.set_height(self.camera.get_height() * scaling_factor),
                  self.camera.animate.set_width(self.camera.get_width() * scaling_factor))
        self.draw_next_level(animate_each_node=True, run_time=5)
        self.wait(5)




manim_configuration = {"quality": "high_quality", "preview": False, "disable_caching": True, "output_file": 'VerticesSmallPrimel7',
                       # "from_animation_number": 0,
                       # "from_animation_number": 0, "upto_animation_number": 70,
                       "renderer": "opengl", "write_to_movie": True, "show_file_in_browser": False, }
if __name__ == '__main__':
    with tempconfig(manim_configuration):
        scene = HoveringECGraphScene(2063, 7)
        scene.render()
