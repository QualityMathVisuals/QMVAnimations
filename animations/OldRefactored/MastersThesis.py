# from sage.all import *
import numpy as np
from manimlib import *

class AppendixH10Box(Scene):
    def construct(self):
        equations_solution_pairs_Z = [
            (MathTex(r'x^2 + y^2 - 5 = 0'), Tex(r'YES', color=GREEN_B)),
            (MathTex(r'6x^2 -5x + 1 = 0'), Tex(r'NO', color=RED_C)),
            (MathTex(r'x^2 + 23y^2 - 41 = 0'), Tex(r'NO', color=RED_C)),
            (MathTex(r'3x^3 + 4y^3 + 5z^3 = 0'), Tex(r'YES', color=GREEN_B)),
        ]
        equations_solution_pairs_Q = [
            (MathTex(r'x^2 + y^2 - 3 = 0'), Tex(r'NO', color=RED_C)),
            (MathTex(r'6x^2 -5x + 1 = 0'), Tex(r'YES', color=GREEN_B)),
            (MathTex(r'x^2 + 23y^2 - 41 = 0'), Tex(r'YES', color=GREEN_B)),
            (MathTex(r'3x^3 + 4y^3 + 5z^3 = 0'), Tex(r'NO', color=RED_C)),
        ]
        H10BoxZ = VGroup(
            Rectangle(width=4, height=2),
            MathTex(r'\textrm{Has}\_\mathbb Z\_\textrm{Solution}').shift(UP * 0.5),
            MathTex(r'\dots').shift(DOWN * 0.5),
        )
        H10BoxQ = VGroup(
            Rectangle(width=4, height=2),
            MathTex(r'\textrm{Has}\_\mathbb Q\_\textrm{Solution}').shift(UP * 0.5),
            MathTex(r'\dots').shift(DOWN * 0.5),
        )
        self.add(H10BoxZ)

        # Process input
        def process_input(H10Box, equation_solution_pair, num_waits=1):
            diophantine_eq = equation_solution_pair[0]
            output = equation_solution_pair[1]
            diophantine_eq.next_to(H10Box, LEFT, buff=1)
            output.next_to(H10Box, RIGHT, buff=1)
            output.save_state()
            output.move_to(H10Box.get_center()).scale(0.01).fade(1)
            self.play(FadeIn(diophantine_eq), run_time=0.5)
            self.wait()
            self.play(diophantine_eq.animate.move_to(H10Box.get_center()).fade(1).scale(0.01))
            for _ in range(num_waits):
                self.play(ApplyWave(H10Box[2]), run_time=0.5)
            self.play(Wiggle(H10Box), output.animate.restore(), run_time=0.5)
            self.play(FadeOut(output), run_time=0.5)

        for pair in equations_solution_pairs_Z:
            process_input(H10BoxZ, pair)

        process_input(H10BoxZ, (MathTex(r'x^{27}y^{15}z^3-y^{100}x^2 + wyx = 0').scale(0.5), Tex(r'NO', color=RED_C)), num_waits=4)

        self.play(ReplacementTransform(H10BoxZ, H10BoxQ))

        for pair in equations_solution_pairs_Q:
            process_input(H10BoxQ, pair)

        process_input(H10BoxQ, (MathTex(r'x^{27}y^{15}z^3-y^{100}x^2 + wyz = 0').scale(0.5), Tex(r'NO', color=GREEN_B)), num_waits=4)


# class B3H10OverFields(ThreeDScene):
#     def construct(self):
#         example_equation = MathTex(r'\textrm{Ex. } 3y^2 + 4x^5 - 1 = 0')
#         X = Y = 3
#         Zgrid = NumberPlane(
#             x_range=[-X, X, 1],
#             y_range=[-Y, Y, 1],
#             x_length=4.2,
#             y_length=4.2,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 3.5,
#             },
#             axis_config={"include_numbers": True},
#         )
#         Qgrid = Zgrid.copy()
#         Rgrid = Zgrid.copy()
#         grids = [
#             Zgrid, Qgrid, Rgrid
#         ]
#         labels = [
#             MathTex(r'\mathbb Z'),
#             MathTex(r'\mathbb Q'),
#             MathTex(r'\mathbb R'),
#             MathTex(r'\mathbb C')
#         ]
#         reasons = [
#             Tex(r'Negative Answer\\DPRM Theorem', color=RED_B).scale(0.8),
#             Tex(r'???'),
#             Tex(r'Positive Answer\\Calculus and Resultants', color=GREEN_C).scale(0.8),
#             Tex(r': Fundamental Theorem of Algebra', color=GREEN_C).scale(0.8)
#         ]

#         self.add_fixed_in_frame_mobjects(*grids, *labels, *reasons, example_equation)
#         self.remove(*reasons, *labels)
#         example_equation.to_edge(UP).to_edge(LEFT)
#         Zgrid.to_edge(LEFT).shift(DOWN * 0.5)
#         Qgrid.next_to(Zgrid, RIGHT)
#         Rgrid.next_to(Qgrid, RIGHT)
#         zgraph = Zgrid.plot_implicit_curve(lambda x, y: 3*y ** 2 + 4*x ** 5 - 1, color=YELLOW)
#         qgraph = Qgrid.plot_implicit_curve(lambda x, y: 3*y ** 2 + 4*x ** 5 - 1, color=YELLOW)
#         rgraph = Rgrid.plot_implicit_curve(lambda x, y: 3*y ** 2 + 4*x ** 5 - 1, color=YELLOW)
#         graphs = [
#             zgraph, qgraph, rgraph
#         ]
#         labels[0].next_to(Zgrid, UP)
#         labels[1].next_to(Qgrid, UP)
#         labels[2].next_to(Rgrid, UP)
#         labels[3].next_to(example_equation, RIGHT, buff=1)
#         reasons[0].next_to(Zgrid, DOWN)
#         reasons[1].next_to(Qgrid, DOWN)
#         reasons[2].next_to(Rgrid, DOWN)
#         reasons[3].next_to(labels[3], RIGHT)
#         integer_dots = []
#         x = y = 0
#         dx = 0
#         dy = -1
#         for i in range(max(2*X + 1, 2*Y + 1) ** 2):
#             if (-X <= x <= X) and (-Y <= y <= Y):
#                 integer_dots.append(
#                     Dot(point=Zgrid.c2p(x, y), radius=0.1, color=WHITE)
#                 )
#             if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
#                 dx, dy = -dy, dx
#             x, y = x + dx, y + dy

#         # integer_dots = VGroup(*integer_dots)
#         # self.play(Write(labels[0]))
#         # self.play(Create(graphs[0]))
#         # total_run_time = 5
#         # self.play(LaggedStart(*[
#         #     Succession(Create(integer_dots[i]), FadeOut(integer_dots[i]), run_time=0.4)
#         #     for i in range(len(integer_dots))
#         # ], run_time=total_run_time, lag_ratio=4/len(integer_dots)))
#         # self.play(Write(reasons[0]))
#         # self.wait()
#         # self.play(Write(labels[3]))
#         # self.play(Write(reasons[3]))
#         # self.wait()
#         # self.play(Write(labels[2]))
#         # self.play(Create(graphs[2]))
#         # self.play(Write(reasons[2]))
#         # self.wait()
#         # self.play(Write(labels[1]))
#         # self.play(Create(graphs[1]))
#         # self.play(Write(reasons[1]))
#         self.add(*reasons, *labels, *graphs)

# class B4ZoomOnCurve(MovingCameraScene):
#     def construct(self):
#         exponent_res_max = 8
#         resolution = 1 / 2
#         grid = NumberPlane(
#             x_range=[-3, 3, 1],
#             y_range=[-3, 3, 1],
#             x_length=7,
#             y_length=7,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 3.5,
#             },
#             axis_config={"include_numbers": True,},
#         )

#         rat_pts_in_range = []
#         centered_pt = [0.795, 0.25]
#         zoom_pt = grid.c2p(*centered_pt)
#         orginal_frame = Rectangle(
#             width=self.camera.frame_width,
#             height=self.camera.frame_height
#         )
#         def frame_at_res(i):
#             return Rectangle(
#                 width=1 * (resolution ** i),
#                 height=1 * (resolution ** i)
#             ).move_to(zoom_pt * (1 - resolution/(i + 1)))

#         grids_many_rat_pts = []
#         for i in range(exponent_res_max):
#             grids_many_rat_pts.append(
#                 NumberPlane(
#                     x_range=[-3, 3, resolution ** (i + 1)],
#                     y_range=[-3, 3, resolution ** (i + 1)],
#                     x_length=7,
#                     y_length=7,
#                     tips=False,
#                     background_line_style={
#                         "stroke_width": 3.5 * (resolution ** (i + 1)),
#                     },
#                     axis_config={"include_numbers": False},
#                 )
#             )
#             rat_pts_this_res = []
#             if i == 0:
#                 frame = orginal_frame
#             else:
#                 frame = frame_at_res(i)
#             for j in range(int(-3 / (resolution ** (i + 1))), int(3 / (resolution ** (i + 1)))):
#                 x = grid.c2p(j * resolution ** (i + 1), 0)[0]
#                 if frame.get_center()[0] - (frame.width) < x < frame.get_center()[0] + (frame.width):
#                     for k in range(int(-3 / (resolution ** (i + 1))), int(3 / (resolution ** (i + 1)))):
#                         y = grid.c2p(0, k * resolution ** (i + 1))[1]
#                         if frame.get_center()[1] - (frame.height) < y < frame.get_center()[1] + (frame.height):
#                             rat_pts_this_res.append(
#                                 Dot(
#                                     point=np.array([x, y, 0]),
#                                     color=PINK,
#                                     z_index=2,
#                                     radius=0.06 * (resolution ** (i + 1))
#                                 )
#                             )
#             rat_pts_in_range.append(VGroup(*rat_pts_this_res))
#         self.add(grid)
#         curve = grid.plot_implicit_curve(lambda x, y: 3*y ** 2 + 4*x ** 5 - 1, color=YELLOW)
#         curve.z_index = 10
#         self.play(Create(curve))
#         self.wait()
#         self.play(Create(grids_many_rat_pts[0]))
#         self.play(Create(rat_pts_in_range[0]))
#         self.wait()
#         for i in range(exponent_res_max - 1):
#             desired_frame = frame_at_res(i)
#             self.play(
#                 self.camera.auto_zoom(desired_frame),
#                 FadeIn(grids_many_rat_pts[i + 1]),
#                 FadeIn(rat_pts_in_range[i + 1]),
#                 curve.animate.set_stroke_width(3.5 * (resolution ** (i + 1))),
#                 run_time=3,
#                 rate_func=rate_functions.linear
#             )
#         self.wait()
#         self.play(self.camera.auto_zoom(orginal_frame), run_time=2)
#         self.play(curve.animate.set_stroke_width(3.5))
#         self.wait()

# class B5PlaneSmoothCurve(ThreeDScene):
#     def construct(self):
#         #Trace out a 3D curve that has a well defined tangent line,
#         #Trace out one that does not.
#         grid = NumberPlane(
#             x_range=[-3, 3, 1],
#             y_range=[-3, 3, 1],
#             x_length=3.5,
#             y_length=3.5,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 3.5,
#             },
#             axis_config={"include_numbers": False},
#         ).to_corner(LEFT + UP)
#         smooth_grids = [
#             grid.copy(),
#             grid.copy().next_to(grid, RIGHT),
#             grid.copy().next_to(grid, DOWN)
#         ]
#         smooth_grids.append(grid.copy().next_to(smooth_grids[2], RIGHT))
#         smooth_examples = [
#             smooth_grids[0].plot_implicit_curve(lambda x, y: y**2 - x**3, color=YELLOW),
#             smooth_grids[1].plot_implicit_curve(lambda x, y: y**2 - x**3 - x**2, color=YELLOW),
#             smooth_grids[2].plot_implicit_curve(lambda x, y: y**3 - 2*x**2 + 1, color=YELLOW),
#             smooth_grids[3].plot_implicit_curve(lambda x, y: y**2 + x**2 - 2, color=YELLOW),
#         ]
#         singular_points = [
#             Dot(point=smooth_grids[0].c2p(0, 0), color=RED_C, radius=.12),
#             Dot(point=smooth_grids[1].c2p(0, 0), color=RED_C, radius=.12)
#         ]
#         labels = [
#             Tex(r'$\times$', r' Singular', color=RED_B).next_to(smooth_grids[1], RIGHT),
#             Tex(r'$\checkmark$', r' Smooth', color=GREEN).next_to(smooth_grids[3], RIGHT),
#         ]
#         labels[0][0].scale(1.5)
#         labels[1][0].scale(1.5)
#         self.add_fixed_in_frame_mobjects(*smooth_grids, *labels, *singular_points)
#         self.remove(*singular_points, *labels)
#         singular_points[0].z_index = 2
#         singular_points[1].z_index = 2
#         self.play(Create(smooth_examples[0]), Create(smooth_examples[1]), Create(smooth_examples[2]), Create(smooth_examples[3]))
#         self.play(Create(singular_points[0]))
#         self.play(Create(singular_points[1]))
#         self.play(Write(labels[0]))
#         self.play(Write(labels[1]))
#         self.wait(5)

#         axes = ThreeDAxes(
#             x_range=[-3, 3, 1],
#             y_range=[-3, 3, 1],
#             z_range=[-3, 3, 1],
#             x_length=3.5,
#             y_length=3.5,
#             z_length=3.5,
#             axis_config={
#                 "include_numbers": False,
#                 "tip_width": 0.15,
#                 "tip_height": 0.15
#             }
#         ).to_corner(LEFT + UP)
#         higher_dimension_examples = [
#             axes.copy(),
#             axes.copy().next_to(axes, RIGHT),
#             axes.copy().next_to(axes, DOWN),
#         ]
#         higher_dimension_examples.append(axes.copy().next_to(higher_dimension_examples[2], RIGHT))
#         def C1(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t < 1:
#                 x = 2 * (t - 1)
#                 y = 2 * (t - 1)
#                 z = 1.5 + (t - 1) ** 3
#             if 1 <= t < 2:
#                 x = np.sin(2 * PI * (t - 1))
#                 y = np.sin(2 * PI * (t - 1))
#                 z = 3 * (1 - t) + 1.5
#             if 2 <= t <= 3:
#                 x = -2 * (t - 2)
#                 y = -2 * (t - 2)
#                 z = -1.5 - (t - 2) ** 3
#             return np.array([x, y, z])

#         def C2(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t < 1:
#                 x = (t - 1) + 2
#                 y = 3 * (t - 1)
#                 z = 3 * (t - 1)**2
#             if 1 <= t < 3:
#                 x = 2 * np.cos(PI * (t - 1))
#                 y = 2 * np.sin(PI * (t - 1))
#                 z = 0
#             if 3 <= t <= 4:
#                 x = (t - 3) + 2
#                 y = 3 * (t - 3)
#                 z = 3 * (t - 3) ** 2
#             return np.array([x, y, z])

#         def C3(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t < 1:
#                 x = 2 * (t - 1)
#                 y = 2 * (t - 1)
#                 z = 1.5 + (t - 1)**3
#             if 1 <= t < 2:
#                 x = np.sin(2 * PI * (t - 1))
#                 y = np.sin(2 * PI * (t - 1))
#                 z = 3 * (1 - t) + 1.5
#             if 2 <= t <= 3:
#                 x = 2 * (t - 2)
#                 y = 2 * (t - 2)
#                 z = -1.5 - (t - 2)**3
#             return np.array([x, y, z])

#         def C4(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t < 1:
#                 x = (t - 1) + 2
#                 y = 3 * (t - 1)
#                 z = 3 * (t - 1)**2 + 1
#             if 1 <= t < 3:
#                 x = 2 * np.cos(PI * (t - 1))
#                 y = 2 * np.sin(PI * (t - 1))
#                 z = 2 - t
#             if 3 <= t <= 4:
#                 x = (t - 3) + 2
#                 y = 3 * (t - 3)
#                 z = -3 * (t - 3) ** 2 - 1
#             return np.array([x, y, z])
#         for ax in higher_dimension_examples:
#             ax.rotate(PI / 3, LEFT)
#             ax.rotate(PI / 6, DOWN)
#         higher_dimension_curves = [
#             higher_dimension_examples[0].plot_parametric_curve(C1, t_range=[0, 3], color=YELLOW),
#             higher_dimension_examples[1].plot_parametric_curve(C2, t_range=[0, 4], color=YELLOW),
#             higher_dimension_examples[2].plot_parametric_curve(C3, t_range=[0, 3], color=YELLOW),
#             higher_dimension_examples[3].plot_parametric_curve(C4, t_range=[0, 4], color=YELLOW)
#         ]
#         self.play(FadeOut(*smooth_grids, *smooth_examples, *singular_points))
#         self.play(FadeIn(*higher_dimension_examples))
#         self.play(Create(higher_dimension_curves[0]))
#         self.play(Create(higher_dimension_curves[1]))
#         self.play(Create(higher_dimension_curves[2]))
#         self.play(Create(higher_dimension_curves[3]))
#         self.wait(5)

# class B5PlaneGenusCurve(Scene):
#     def construct(self):
#         grid1 = NumberPlane(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=5,
#             y_length=5,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 3.5,
#             },
#             axis_config={"include_numbers": False},
#         ).to_edge(LEFT)
#         grid2 = grid1.copy().next_to(grid1, RIGHT)
#         self.add(grid1, grid2)
#         equations = [
#             MathTex(r'y - x^4 - 3x^2 - 2').next_to(grid1, UP),
#             MathTex(r'y^4 + x^2y^2 + x').next_to(grid2, UP),
#             MathTex(r'y^3 + x^3 + 1').next_to(grid1, UP),
#             MathTex(r'y^4x^4 +x^3y + y^3 + x^3').next_to(grid2, UP),
#         ]
#         answers = [
#             Tex(r'Genus 0').next_to(grid1, DOWN),
#             Tex(r'Genus 2').next_to(grid2, DOWN),
#             Tex(r'Genus 1').next_to(grid1, DOWN),
#             Tex(r'Genus 9').next_to(grid2, DOWN),
#         ]
#         # curves_ex = [
#         #     grid1.plot_implicit_curve(lambda x, y: x ** 5 + 3*x ** 2 + 2, color=YELLOW),
#         #     grid2.plot_implicit_curve(lambda x, y: y**3 + x**3 + 1, color=YELLOW),
#         #     grid1.plot_implicit_curve(lambda x, y: y**4 + x**2*y**2 + x, color=YELLOW),
#         #     grid2.plot_implicit_curve(lambda x, y: y**5*x**5 + x*y**2 + y**3 + x**3, color=YELLOW),
#         # ]
#         peg_coords = [
#             [[0, 0], [0, 1], [4, 0], [2, 0]],
#             [[0, 4], [2, 2], [1, 0]],
#             [[0, 0], [0, 3], [3, 0]],
#             [[0, 3], [3, 2], [4, 4], [3, 0]],
#         ]
#         pegs = []
#         for i in range(len(peg_coords)):
#             coords = peg_coords[i]
#             peg_grp = []
#             for coord in coords:
#                 if i % 2 == 0:
#                     peg_grp.append(Dot(point=grid1.c2p(*coord), color=PINK, radius=0.1))
#                 else:
#                     peg_grp.append(Dot(point=grid2.c2p(*coord), color=PINK, radius=0.1))
#             pegs.append(VGroup(*peg_grp))

#         bands = [
#             Circle(arc_center=grid1.c2p(2.5, 0), radius=1.5, color=PINK).flip(),
#             Circle(arc_center=grid2.c2p(1.5, 1.5), radius=1.5, color=PINK).flip(),
#             Circle(arc_center=grid1.c2p(1.5, 1.5), radius=1.5, color=PINK).flip(),
#             Circle(arc_center=grid2.c2p(2.5, 2.5), radius=2, color=PINK).flip()
#         ]
#         tight_bands = [
#             Polygon(grid1.c2p(*peg_coords[0][0]), grid1.c2p(*peg_coords[0][1]), grid1.c2p(*peg_coords[0][2]), color=PINK),
#             Polygon(grid2.c2p(*peg_coords[1][0]), grid2.c2p(*peg_coords[1][1]), grid2.c2p(*peg_coords[1][2]),
#                     color=PINK),
#             Polygon(grid1.c2p(*peg_coords[2][0]), grid1.c2p(*peg_coords[2][1]), grid1.c2p(*peg_coords[2][2]), color=PINK),
#             Polygon(grid2.c2p(*peg_coords[3][0]), grid2.c2p(*peg_coords[3][2]), grid2.c2p(*peg_coords[3][3]), color=PINK),
#         ]
#         highlighted_points = [
#             VGroup(),
#             VGroup(
#                 Dot(point=grid2.c2p(1, 1), color=RED_B),
#                 Dot(point=grid2.c2p(1, 2), color=RED_B)
#             ),
#             VGroup(
#                 Dot(point=grid1.c2p(1, 1), color=RED_B)
#             ),
#             VGroup(
#                 Dot(point=grid2.c2p(1, 3), color=RED_B),
#                 Dot(point=grid2.c2p(2, 3), color=RED_B),
#                 Dot(point=grid2.c2p(2, 2), color=RED_B),
#                 Dot(point=grid2.c2p(3, 3), color=RED_B),
#                 Dot(point=grid2.c2p(3, 2), color=RED_B),
#                 Dot(point=grid2.c2p(3, 1), color=RED_B),
#             )
#         ]
#         for i in [0, 1]:
#             self.play(Write(equations[i]))
#             self.play(Create(pegs[i]))
#             self.play(Create(bands[i]))
#             self.play(ReplacementTransform(bands[i], tight_bands[i]))
#             self.play(Create(highlighted_points[i]))
#             self.play(Write(answers[i]))
#         self.play(FadeOut(equations[0], equations[1], pegs[0], pegs[1], bands[0], bands[1], tight_bands[0], tight_bands[1], highlighted_points[0], highlighted_points[1], answers[0], answers[1]))
#         for i in [2, 3]:
#             self.play(Write(equations[i]))
#             self.play(Create(pegs[i]))
#             self.play(Create(bands[i]))
#             self.play(ReplacementTransform(bands[i], tight_bands[i]))
#             self.play(Create(highlighted_points[i]))
#             self.play(Write(answers[i]))

# class B18SmoothCurveTwist(ThreeDScene):
#     def construct(self):
#         self.set_camera_orientation(phi=PI / 3)
#         self.begin_ambient_camera_rotation(0.15)
#         axes = ThreeDAxes(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             z_range=[-3, 3, 1],
#             x_length=10,
#             y_length=10,
#             z_length=6,
#             axis_config={
#                 "numbers_to_include": range(-4, 5, 2),
#                 "tip_width": 0.15,
#                 "tip_height": 0.15
#             },
#             z_axis_config={
#                 "numbers_to_include": range(-2, 3, 2)
#             }
#         )

#         def C1(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 x = t
#                 z = np.sqrt(1 - x ** 2)
#                 y = np.sqrt(x*z)
#             if 1 < t <= 2:
#                 x = 2 - t
#                 z = np.sqrt(1 - x ** 2)
#                 y = -np.sqrt(x * z)
#             if 2 < t <= 3:
#                 x = 2 - t
#                 z = -np.sqrt(1 - x ** 2)
#                 y = np.sqrt(x * z)
#             if 3 < t <= 4:
#                 x = t - 4
#                 z = -np.sqrt(1 - x ** 2)
#                 y = -np.sqrt(x * z)
#             return np.array([y, x, z])

#         def C2(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 x = 2.5 * (1 - t)
#                 z = np.sqrt(x ** 2 + 1)
#                 y = np.sqrt(x)
#             if 1 < t <= 2:
#                 x = 2.5 * (t - 1)
#                 z = np.sqrt(x ** 2 + 1)
#                 y = -np.sqrt(x)
#             if 2 < t <= 3:
#                 x = 2.5 * (3 - t)
#                 z = -np.sqrt(x ** 2 + 1)
#                 y = -np.sqrt(x)
#             if 3 < t <= 4:
#                 x = 2.5 * (t - 3)
#                 z = -np.sqrt(x ** 2 + 1)
#                 y = np.sqrt(x)
#             return np.array([-y, -z, x])

#         def C3(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 x = 5 * t
#                 y = 1 / x
#                 z = np.sqrt((x ** 2 + y ** 2) / 3)
#             if 1 < t <= 2:
#                 x = 5 * (1 - t)
#                 y = 1 / x
#                 z = np.sqrt((x ** 2 + y ** 2) / 3)
#             if 2 < t <= 3:
#                 x = 5 * (t - 2)
#                 y = 1 / x
#                 z = -np.sqrt((x ** 2 + y ** 2) / 3)
#             if 3 < t <= 4:
#                 x = 5 * (3 - t)
#                 y = 1 / x
#                 z = -np.sqrt((x ** 2 + y ** 2) / 3)
#             return np.array([-y, x, z])

#         def C4(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 x = 3 * (1 - t)
#                 z = 2 * (t - 1) - 1
#                 y = np.sqrt(x**3 + x**2)
#             if 1 < t <= 2:
#                 x = (1 - t)
#                 z = (t - 1) - 1
#                 y = -np.sqrt(x**3 + x**2)
#             if 2 < t <= 3:
#                 x = (t - 3)
#                 z = (t - 2)
#                 y = np.sqrt(x ** 3 + x ** 2)
#             if 3 < t <= 4:
#                 x = 3 * (t - 3)
#                 z = 2 * (t - 3) + 1
#                 y = -np.sqrt(x ** 3 + x ** 2)
#             return np.array([x, y, z])

#         def C5(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 z = -5 * t + 3
#                 y = np.sqrt(z + 2)
#                 x = np.sqrt(z**2 + 1)
#             if 1 < t <= 2:
#                 z = -2 + (t - 1) * 5
#                 y = -np.sqrt(z + 2)
#                 x = np.sqrt(z ** 2 + 1)
#             if 2 < t <= 3:
#                 z = -5 * (t - 2) + 3
#                 y = -np.sqrt(z + 2)
#                 x = -np.sqrt(z ** 2 + 1)
#             if 3 < t <= 4:
#                 z = -2 + (t - 3) * 5
#                 y = np.sqrt(z + 2)
#                 x = -np.sqrt(z ** 2 + 1)
#             return np.array([y, x, z])

#         def C6(t):
#             x, y, z = 0, 0, 0
#             if 0 <= t <= 1:
#                 y = -1 * t
#                 z = np.cbrt(y)
#                 x = np.sqrt(y - z)
#             if 1 < t <= 2:
#                 y = (t - 2)
#                 z = np.cbrt(y)
#                 x = -np.sqrt(y - z)
#             if 2 < t <= 3:
#                 y = 4 * (3 - t) + 1
#                 z = np.cbrt(y)
#                 x = -np.sqrt(y - z)
#             if 3 < t <= 4:
#                 y = 4 * (t - 3) + 1
#                 z = np.cbrt(y)
#                 x = np.sqrt(y - z)
#             return np.array([x, y, z])

#         def create_R3_examples():
#             return [
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C1,
#                         color=YELLOW,
#                         t_range=[0, 2, 0.01],
#                     ).set_shade_in_3d(True),
#                     axes.plot_parametric_curve(
#                         C1,
#                         color=ORANGE,
#                         t_range=[2.01, 4, 0.01],
#                     ).set_shade_in_3d(True)
#                 ),
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C2,
#                         color=YELLOW,
#                         t_range=[0, 2, 0.01],
#                     ).set_shade_in_3d(True),
#                     axes.plot_parametric_curve(
#                         C2,
#                         color=ORANGE,
#                         t_range=[2.01, 4, 0.01],
#                     ).set_shade_in_3d(True)
#                 ),
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C3,
#                         color=YELLOW,
#                         t_range=[0.1, 2 - 0.1, 0.01],
#                         discontinuities=[1],
#                         dt=0.1,
#                     ).set_shade_in_3d(True),
#                     axes.plot_parametric_curve(
#                         C3,
#                         color=ORANGE,
#                         t_range=[2 + 0.1, 4 - 0.1, 0.01],
#                         discontinuities=[3],
#                         dt=0.1,
#                     ).set_shade_in_3d(True)
#                 ),
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C4,
#                         color=YELLOW,
#                         t_range=[0, 4, 0.01]
#                     ).set_shade_in_3d(True)
#                 ),
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C5,
#                         color=YELLOW,
#                         t_range=[0, 2, 0.01]
#                     ).set_shade_in_3d(True),
#                     axes.plot_parametric_curve(
#                         C5,
#                         color=ORANGE,
#                         t_range=[2.01, 4, 0.01]
#                     ).set_shade_in_3d(True)
#                 ),
#                 VGroup(
#                     axes.plot_parametric_curve(
#                         C6,
#                         color=YELLOW,
#                         t_range=[0, 2, 0.01],
#                     ).set_shade_in_3d(True),
#                     axes.plot_parametric_curve(
#                         C6,
#                         color=ORANGE,
#                         t_range=[2.01, 4, 0.01],
#                     ).set_shade_in_3d(True)
#                 )
#             ]
#         examples_R3 = create_R3_examples()

#         examples_R2_params = [
#             [
#                 [lambda x, y: x ** 4 + y ** 4 - y ** 2, YELLOW, [0, 1.1]],
#                 [lambda x, y: x ** 4 + y ** 4 - y ** 2, ORANGE, [-1.1, 0]]
#             ],
#             [
#                 [lambda x, y: y ** 2 - x ** 4 - 1, YELLOW, [-4, 0]],
#                 [lambda x, y: y ** 2 - x ** 4 - 1, ORANGE, [0, 4]]
#             ],
#             [
#                 [lambda x, y: 3 * x**2*y**2 - 1 - y**4, YELLOW, [-5, 0]],
#                 [lambda x, y: 3 * x**2*y**2 - 1 - y**4, ORANGE, [0, 5]]
#             ],
#             [
#                 [lambda x, y: y ** 2 - x ** 3 - x ** 2, YELLOW, [-5, 5]]
#             ],
#             [
#                 [lambda x, y: y**2 - (x**2 - 2)**2-1, YELLOW, [0, 5]],
#                 [lambda x, y: y**2 - (x**2 - 2)**2-1, ORANGE, [-5, 0]]
#             ],
#             [
#                 [lambda x, y: y - (y - x**2)**3, YELLOW, [-1, 0.1]],
#                 [lambda x, y: y - (y - x**2)**3, ORANGE, [0.9, 5]]
#             ]
#         ]
#         examples_R2_embedded = []
#         for param in examples_R2_params:
#             curve = []
#             for packet in param:
#                 curve.append(
#                     ImplicitFunction(packet[0], color=packet[1], y_range=packet[2])
#                 )
#             curveGp = VGroup(*curve)
#             examples_R2_embedded.append(curveGp)
#         labels_R2 = [
#             MathTex(r'x^4 + y^4 - y^2'),
#             MathTex(r'y^2 - x^4 - 1'),
#             MathTex(r'3x^2y^2 - 1 - y^4'),
#             MathTex(r'y^2 - x^3 - x^2'),
#             MathTex(r'y^2 - (x^2 - 2)^2-1'),
#             MathTex(r'y - (y - x^2)^3'),
#         ]

#         singular_points = [
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.07),
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.00),
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.00),
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.07),
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.00),
#             Dot3D(point=ORIGIN, color=RED_E, radius=0.00),
#         ]

#         self.add(axes)
#         #B8 Anims
#         # for i in range(6):
#         #     self.play(Create(examples_R3[i]))
#         #     self.play(ReplacementTransform(examples_R3[i], examples_R2_embedded[i]))
#         #     self.play(FadeIn(singular_points[i]))
#         #     self.play(FadeOut(singular_points[i], examples_R2_embedded[i], examples_R3[i]))
#         # self.wait()
#         plane = NumberPlane(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=5,
#             y_length=5,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 2,
#             },
#             axis_config={"numbers_to_include": range(-4, 5, 2)},
#         )

#         self.add_fixed_in_frame_mobjects(plane)
#         plane.to_edge(RIGHT)
#         #B10 Modification
#         # self.play(FadeIn(plane))
#         examples_R2_plane = []

#         for param in examples_R2_params:
#             curve = []
#             for packet in param:
#                 curve.append(plane.plot_implicit_curve(packet[0], color=YELLOW))
#             curveGp = VGroup(*curve)
#             self.add_fixed_in_frame_mobjects(curveGp)
#             self.remove(curveGp)
#             examples_R2_plane.append(curveGp)
#         self.add_fixed_in_frame_mobjects(*labels_R2)
#         for lbl in labels_R2:
#             lbl.next_to(plane, UP)
#         self.remove(*labels_R2)

#         #B10 question
#         ec_example_param = [
#             [
#                 lambda x, y: y ** 2 - x ** 3 + 4 * x, [-5, 5]
#             ],
#             [
#                 lambda x, y: y ** 2 - x ** 3 - (81 / 4) * x, [-5, 5]
#             ],
#             [
#                 lambda x, y: -y ** 2 + x ** 3 - 6 * x, [-5, 5]
#             ],
#             [
#                 lambda x, y: y ** 2 - x ** 3 - x, [-5, 5]
#             ],
#             [
#                 lambda x, y: -y ** 2 + x ** 3 + (704 / 125) * x ** 2 + (12288 / 3125) * x + (262144 / 390625), [-5, 5]
#             ],
#             [
#                 lambda x, y: y ** 2 - x ** 3 + x, [-5, 5]
#             ]
#         ]

#         ec_examples = []
#         for param in ec_example_param:
#             curv = plane.plot_implicit_curve(param[0], color=YELLOW)
#             self.add_fixed_in_frame_mobjects(curv)
#             self.remove(curv)
#             ec_examples.append(curv)

#         question_mark = Tex(r'?').scale(3)
#         mysteryEquation = MathTex(r'y^2 = x^3 + Ax + B')
#         self.add_fixed_in_frame_mobjects(question_mark, mysteryEquation)
#         question_mark.move_to(plane)
#         mysteryEquation.next_to(plane, UP)
#         self.remove(question_mark, mysteryEquation)
#         #End B10 stuff

#         self.stop_ambient_camera_rotation()
#         camera_angle = PI / 4.5
#         #B10 Modification
#         self.set_camera_orientation(phi=0, theta=3 * PI / 2)
#         axes.rotate(angle=-camera_angle, axis=RIGHT)
#         axes.shift(LEFT * 3).scale(0.6)
#         # self.move_camera(phi=0, theta=3 * PI / 2, added_anims=[axes.animate.rotate(angle=-camera_angle, axis=RIGHT)])
#         # self.play(axes.animate.shift(LEFT * 3).scale(0.6))
#         arrow_label = DoubleArrow(start=LEFT, end=plane.get_edge_center(LEFT) + LEFT * 0.2, color=BLUE).shift(DOWN)
#         phi_label = MathTex(r'\phi', color=BLUE).next_to(arrow_label, DOWN)
#         phi_1_label = MathTex(r'\phi:C \to ', r'C').to_edge(DOWN, buff=1)
#         phi_2_label = MathTex(r'\mathbb Q', r'-\textrm{isomorphism}').next_to(phi_1_label, DOWN)

#         self.add_fixed_in_frame_mobjects(phi_label, arrow_label, phi_1_label, phi_2_label)
#         self.remove(phi_label, arrow_label, phi_1_label, phi_2_label)
#         self.play(Write(arrow_label), Write(phi_label))
#         self.play(Write(phi_1_label), Write(phi_2_label))
#         self.wait()
#         # examples_R3 = create_R3_examples()
#         # for i in range(6):
#         #     self.play(Create(examples_R3[i]))
#         #     self.play(FadeTransform(examples_R3[i], examples_R2_plane[i]), Write(labels_R2[i]))
#         #     #B10 modification
#         #     self.play(ReplacementTransform(examples_R2_plane[i], ec_examples[i]), TransformMatchingTex(labels_R2[i], mysteryEquation), FadeIn(question_mark))
#         #     self.play(FadeOut(ec_examples[i], mysteryEquation, question_mark))
#         #     # self.play(FadeOut(examples_R2_plane[i]), FadeOut(labels_R2[i]))
#         #
#         # self.wait()

#         #EC examples B8
#         phi_1_label_e = MathTex(r'\phi:C \to ', r'E').to_edge(DOWN, buff=1)
#         phi_2_labels = [
#             MathTex(r'\mathbb Q', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#             MathTex(r'\mathbb Q', r'(\sqrt[4]{-1})', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#             MathTex(r'\mathbb Q', r'(\sqrt{3}, \sqrt{2})', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#             MathTex(r'\mathbb Q', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#             MathTex(r'\mathbb Q', r'(\sqrt{5})', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#             MathTex(r'\mathbb Q', r'-\textrm{isomorphism}').next_to(phi_1_label_e, DOWN),
#         ]
#         twist_labels = [
#             MathTex(r'y^2 = x^3 - 4x'),
#             MathTex(r'y^2 = x^3 + \frac{81}{4}x'),
#             MathTex(r'y^2 = x^3 + 1296x'),
#             MathTex(r'y^2 = x^3 + x^2'),
#             MathTex(r'y^2 = x^3 + \frac{704}{125}x^2 + \frac{12288}{3125}x + \frac{262144}{390625}').scale(0.7),
#             MathTex(r'y^2 = x^3 - x'),
#         ]
#         for twist_label in twist_labels:
#             twist_label.next_to(plane, UP)
#         self.add_fixed_in_frame_mobjects(phi_1_label_e, *phi_2_labels)
#         self.remove(phi_1_label_e, *phi_2_labels)
#         self.play(TransformMatchingTex(phi_1_label, phi_1_label_e))
#         twist_example_param = [
#             [
#                 lambda x, y: y ** 2 - x ** 3 + 4 * x, [-5, 5]
#             ],
#             [
#                 lambda x, y: y**2 - x**3 - (81/4)*x, [-5, 5]
#             ],
#             [
#                 lambda x, y: -y**2 + x**3 - 1296*x, [-5, 5]
#             ],
#             [
#                 lambda x, y: y ** 2 - x ** 3 - x ** 2, [-5, 5]
#             ],
#             [
#                 lambda x, y: -y**2 + x**3 + (704/125)*x**2 + (12288/3125)*x + (262144/390625), [-5, 5]
#             ],
#             [
#                 lambda x, y: y**2 - x**3 + x, [-5, 5]
#             ]
#         ]

#         twist_examples = []
#         for param in twist_example_param:
#             curv = plane.plot_implicit_curve(param[0], color=YELLOW)
#             self.add_fixed_in_frame_mobjects(curv)
#             self.remove(curv)
#             twist_examples.append(curv)
#         examples_R3 = create_R3_examples()
#         self.play(TransformMatchingTex(phi_1_label, phi_1_label_e), TransformMatchingTex(phi_2_label, phi_2_labels[0]))
#         self.play(Create(examples_R3[0]))
#         self.play(FadeTransform(examples_R3[0], twist_examples[0]), Write(twist_labels[0]))
#         self.wait()
#         self.play(FadeOut(twist_examples[0], twist_labels[0]))
#         for i in range(1, 6):
#             self.play(Create(examples_R3[i]))
#             self.play(
#                 FadeTransform(examples_R3[i], twist_examples[i]),
#                 Write(twist_labels[i]),
#                 TransformMatchingTex(phi_2_labels[i - 1], phi_2_labels[i])
#             )
#             if len(phi_2_labels[i]) > 2:
#                 self.play(Circumscribe(phi_2_labels[i][1]))
#             self.wait()
#             self.play(FadeOut(twist_examples[i], twist_labels[i]))

# EllipticCurve1616Points = [
# (0.00000 , 4.0000 , 1.0000),
# (4.0000 , 4.0000 , 1.0000),
# (-4.0000 , -4.0000 , 1.0000),
# (8.0000 , -20.000 , 1.0000),
# (1.0000 , -1.0000 , 1.0000),
# (24.000 , 116.00 , 1.0000),
# (-2.2222 , 6.3704 , 1.0000),
# (3.3600 , -0.41590 , 1.0000),
# (-1.6327 , -6.1458 , 1.0000),
# (40.250 , -254.12 , 1.0000),
# (0.87713 , 1.6251 , 1.0000),
# (6.4542 , 13.476 , 1.0000),
# (-4.2988 , 2.3111 , 1.0000),
# (4.4531 , -5.7495 , 1.0000),
# (0.34018 , -3.2552 , 1.0000),
# (454.54 , 9690.4 , 1.0000),
# (-0.41027 , 4.7429 , 1.0000),
# (3.6893 , 2.6805 , 1.0000),
# (-3.5614 , -5.2737 , 1.0000),
# (10.342 , -30.931 , 1.0000),
# (1.0656 , -0.40111 , 1.0000),
# (15.993 , 62.056 , 1.0000),
# (-2.8162 , 6.2228 , 1.0000),
# (3.4392 , -1.2853 , 1.0000),
# (-1.0775 , -5.6558 , 1.0000),
# (81.391 , -733.41 , 1.0000),
# (0.69394 , 2.2872 , 1.0000),
# (5.3984 , 9.3248 , 1.0000),
# (-4.4255 , 0.36393 , 1.0000),
# (5.1001 , -8.1888 , 1.0000),
# (0.61163 , -2.5383 , 1.0000),
# (113.66 , 1211.0 , 1.0000),
# (-0.88821 , 5.4324 , 1.0000),
# (3.4888 , 1.6263 , 1.0000),
# (-3.0259 , -6.0588 , 1.0000),
# (14.076 , -50.792 , 1.0000),
# (1.0756 , 0.18652 , 1.0000),
# (11.493 , 36.746 , 1.0000),
# (-3.3754 , 5.6169 , 1.0000),
# (3.6049 , -2.2732 , 1.0000),
# (-0.57657 , -5.0034 , 1.0000),
# (244.41 , -3820.6 , 1.0000),
# (0.44672 , 2.9903 , 1.0000),
# (4.6625 , 6.5389 , 1.0000),
# (-4.3660 , -1.6225 , 1.0000),
# (6.0245 , -11.759 , 1.0000),
# (0.81767 , -1.8612 , 1.0000),
# (50.566 , 358.47 , 1.0000),
# (-1.4255 , 5.9926 , 1.0000),
# (3.3794 , 0.72391 , 1.0000),
# (-2.4396 , -6.3651 , 1.0000),
# (20.490 , -91.055 , 1.0000),
# (1.0303 , 0.77992 , 1.0000),
# (8.7371 , 23.306 , 1.0000),
# (-3.8545 , 4.5172 , 1.0000),
# (3.8725 , -3.4805 , 1.0000),
# (-0.14114 , -4.2726 , 1.0000),
# (3435.5 , -2.0137E5 , 1.0000),
# (0.13183 , 3.7273 , 1.0000),
# (4.1461 , 4.5754 , 1.0000),
# (-4.1269 , -3.4270 , 1.0000),
# (7.3658 , -17.256 , 1.0000),
# (0.96207 , -1.2236 , 1.0000),
# (28.518 , 150.84 , 1.0000),
# (-2.0055 , 6.3263 , 1.0000),
# (3.3510 , -0.11375 , 1.0000),
# (-1.8445 , -6.2639 , 1.0000),
# (32.810 , -186.58 , 1.0000),
# (0.92862 , 1.3939 , 1.0000),
# (6.9476 , 15.498 , 1.0000),
# (-4.2086 , 2.9653 , 1.0000),
# (4.2691 , -5.0498 , 1.0000),
# (0.22453 , -3.5240 , 1.0000),
# (1122.8 , 37621. , 1.0000),
# (-0.25298 , 4.4756 , 1.0000),
# (3.7882 , 3.1226 , 1.0000),
# (-3.7345 , -4.8650 , 1.0000),
# (9.3695 , -26.241 , 1.0000),
# (1.0482 , -0.61684 , 1.0000),
# (18.351 , 76.825 , 1.0000),
# (-2.6019 , 6.3258 , 1.0000),
# (3.4009 , -0.96001 , 1.0000),
# (-1.2740 , -5.8580 , 1.0000),
# (61.152 , -477.20 , 1.0000),
# (0.76778 , 2.0416 , 1.0000),
# (5.7385 , 10.637 , 1.0000),
# (-4.4006 , 1.0907 , 1.0000),
# (4.8380 , -7.1992 , 1.0000),
# (0.52066 , -2.7947 , 1.0000),
# (169.79 , 2211.8 , 1.0000),
# (-0.70692 , 5.1921 , 1.0000),
# (3.5504 , 1.9870 , 1.0000),
# (-3.2289 , -5.8307 , 1.0000),
# (12.498 , -42.053 , 1.0000),
# (1.0783 , -0.027063 , 1.0000),
# (12.867 , 44.047 , 1.0000),
# (-3.1788 , 5.8941 , 1.0000),
# (3.5338 , -1.8943 , 1.0000),
# (-0.75160 , -5.2537 , 1.0000),
# (152.34 , -1879.6 , 1.0000),
# (0.54430 , 2.7299 , 1.0000),
# (4.9006 , 7.4351 , 1.0000),
# (-4.4092 , -0.90887 , 1.0000),
# (5.6489 , -10.290 , 1.0000),
# (0.75005 , -2.1027 , 1.0000),
# (65.449 , 528.52 , 1.0000),
# (-1.2240 , 5.8095 , 1.0000),
# (3.4095 , 1.0403 , 1.0000),
# (-2.6560 , -6.3055 , 1.0000),
# (17.711 , -72.723 , 1.0000),
# (1.0533 , 0.56250 , 1.0000),
# (9.5986 , 27.327 , 1.0000),
# (-3.6924 , 4.9736 , 1.0000),
# (3.7619 , -3.0080 , 1.0000),
# (-0.29156 , -4.5432 , 1.0000),
# (858.89 , -25171. , 1.0000),
# (0.25439 , 3.4563 , 1.0000),
# (4.3130 , 5.2174 , 1.0000),
# (-4.2333 , -2.8048 , 1.0000),
# (6.8173 , -14.959 , 1.0000),
# (0.91645 , -1.4514 , 1.0000),
# (34.467 , 201.02 , 1.0000),
# (-1.7910 , 6.2379 , 1.0000),
# (3.3523 , 0.18815 , 1.0000),
# (-2.0597 , -6.3417 , 1.0000),
# (27.271 , -140.93 , 1.0000),
# (0.97230 , 1.1672 , 1.0000),
# (7.5162 , 17.898 , 1.0000),
# (-4.0969 , 3.5757 , 1.0000),
# (4.1076 , -4.4253 , 1.0000),
# (0.099650 , -3.7956 , 1.0000),
# (6119.8 , 4.7874E5 , 1.0000),
# (-0.10488 , 4.2044 , 1.0000),
# (3.9028 , 3.6059 , 1.0000),
# (-3.8926 , -4.3930 , 1.0000),
# (8.5415 , -22.417 , 1.0000),
# (1.0235 , -0.83473 , 1.0000),
# (21.292 , 96.585 , 1.0000),
# (-2.3852 , 6.3713 , 1.0000),
# (3.3736 , -0.64613 , 1.0000),
# (-1.4769 , -6.0340 , 1.0000),
# (47.634 , -327.62 , 1.0000),
# (0.83333 , 1.8015 , 1.0000),
# (6.1268 , 12.164 , 1.0000),
# (-4.3513 , 1.7980 , 1.0000),
# (4.6074 , -6.3316 , 1.0000),
# (0.42085 , -3.0563 , 1.0000),
# (280.70 , 4702.4 , 1.0000),
# (-0.53404 , 4.9388 , 1.0000),
# (3.6247 , 2.3724 , 1.0000),
# (-3.4231 , -5.5370 , 1.0000),
# (11.186 , -35.164 , 1.0000),
# (1.0738 , -0.24054 , 1.0000),
# (14.522 , 53.348 , 1.0000),
# (-2.9739 , 6.1058 , 1.0000),
# (3.4753 , -1.5392 , 1.0000),
# (-0.93494 , -5.4902 , 1.0000),
# (103.97 , -1059.3 , 1.0000),
# (0.63308 , 2.4748 , 1.0000),
# (5.1713 , 8.4589 , 1.0000),
# (-4.4279 , -0.18255 , 1.0000),
# (5.3199 , -9.0247 , 1.0000),
# (0.67411 , -2.3496 , 1.0000),
# (88.049 , 825.36 , 1.0000),
# (-1.0293 , 5.6016 , 1.0000),
# (3.4506 , 1.3692 , 1.0000),
# (-2.8693 , -6.1876 , 1.0000),
# (15.476 , -58.947 , 1.0000),
# (1.0688 , 0.34728 , 1.0000),
# (10.612 , 32.271 , 1.0000),
# (-3.5158 , 5.3660 , 1.0000),
# (3.6668 , -2.5754 , 1.0000),
# (-0.45113 , -4.8090 , 1.0000),
# (381.73 , -7457.9 , 1.0000),
# (0.36774 , 3.1884 , 1.0000),
# (4.5031 , 5.9384 , 1.0000),
# (-4.3178 , -2.1414 , 1.0000),
# (6.3409 , -13.019 , 1.0000),
# (0.86297 , -1.6838 , 1.0000),
# (42.516 , 276.02 , 1.0000),
# (-1.5802 , 6.1105 , 1.0000),
# (3.3639 , 0.49274 , 1.0000),
# (-2.2767 , -6.3738 , 1.0000),
# (23.038 , -108.97 , 1.0000),
# (1.0083 , 0.94450 , 1.0000),
# (8.1744 , 20.771 , 1.0000),
# (-3.9652 , 4.1351 , 1.0000),
# (3.9664 , -3.8649 , 1.0000),
# (-0.034516 , -4.0684 , 1.0000),
# (54644. , -1.2774E7 , 1.0000),
# (0.033930 , 3.9315 , 1.0000),
# (4.0348 , 4.1386 , 1.0000),
# (-4.0336 , -3.8613 , 1.0000),
# (7.8320 , -19.264 , 1.0000),
# (0.99120 , -1.0557 , 1.0000),
# (25.025 , 123.64 , 1.0000),
# (-2.1678 , 6.3638 , 1.0000),
# (3.3568 , -0.33982 , 1.0000),
# (-1.6854 , -6.1789 , 1.0000),
# (38.161 , -234.47 , 1.0000),
# (0.89078 , 1.5666 , 1.0000),
# (6.5715 , 13.952 , 1.0000),
# (-4.2783 , 2.4789 , 1.0000),
# (4.4047 , -5.5662 , 1.0000),
# (0.31204 , -3.3223 , 1.0000),
# (550.34 , 12910. , 1.0000),
# (-0.36997 , 4.6764 , 1.0000),
# (3.7126 , 2.7878 , 1.0000),
# (-3.6061 , -5.1773 , 1.0000),
# (10.083 , -29.661 , 1.0000),
# (1.0619 , -0.45501 , 1.0000),
# (16.538 , 65.381 , 1.0000),
# (-2.7628 , 6.2543 , 1.0000),
# (3.4286 , -1.2024 , 1.0000),
# (-1.1261 , -5.7087 , 1.0000),
# (75.461 , -654.61 , 1.0000),
# (0.71324 , 2.2251 , 1.0000),
# (5.4796 , 9.6360 , 1.0000),
# (-4.4216 , 0.54755 , 1.0000),
# (5.0312 , -7.9280 , 1.0000),
# (0.58965 , -2.6021 , 1.0000),
# (124.78 , 1393.1 , 1.0000),
# (-0.84199 , 5.3735 , 1.0000),
# (3.5031 , 1.7147 , 1.0000),
# (-3.0775 , -6.0077 , 1.0000),
# (13.652 , -48.395 , 1.0000),
# (1.0770 , 0.13304 , 1.0000),
# (11.815 , 38.420 , 1.0000),
# (-3.3270 , 5.6926 , 1.0000),
# (3.5858 , -2.1757 , 1.0000),
# (-0.61965 , -5.0672 , 1.0000),
# (214.74 , -3146.2 , 1.0000),
# (0.47202 , 2.9245 , 1.0000),
# (4.7193 , 6.7526 , 1.0000),
# (-4.3791 , -1.4456 , 1.0000),
# (5.9255 , -11.369 , 1.0000),
# (0.80148 , -1.9212 , 1.0000),
# (53.779 , 393.31 , 1.0000),
# (-1.3744 , 5.9493 , 1.0000),
# (3.3860 , 0.80237 , 1.0000),
# (-2.4941 , -6.3553 , 1.0000),
# (19.733 , -85.932 , 1.0000),
# (1.0368 , 0.72517 , 1.0000),
# (8.9404 , 24.240 , 1.0000),
# (-3.8154 , 4.6375 , 1.0000),
# (3.8433 , -3.3578 , 1.0000),
# (-0.17799 , -4.3408 , 1.0000),
# (2196.1 , -1.0292E5 , 1.0000),
# (0.16343 , 3.6591 , 1.0000),
# (4.1859 , 4.7298 , 1.0000),
# (-4.1555 , -3.2756 , 1.0000),
# (7.2208 , -16.642 , 1.0000),
# (0.95136 , -1.2803 , 1.0000),
# (29.854 , 161.70 , 1.0000),
# (-1.9514 , 6.3081 , 1.0000),
# (3.3503 , -0.034939 , 1.0000),
# (-1.8982 , -6.2874 , 1.0000),
# (31.271 , -173.47 , 1.0000),
# (0.94030 , 1.3366 , 1.0000),
# (7.0826 , 16.061 , 1.0000),
# (-4.1826 , 3.1224 , 1.0000),
# (4.2266 , -4.8865 , 1.0000),
# (0.19408 , -3.5919 , 1.0000),
# (1529.9 , 59842. , 1.0000),
# (-0.21497 , 4.4079 , 1.0000),
# (3.8154 , 3.2395 , 1.0000),
# (-3.7757 , -4.7525 , 1.0000),
# (9.1494 , -25.209 , 1.0000),
# (1.0427 , -0.67115 , 1.0000),
# (19.027 , 81.238 , 1.0000),
# (-2.5478 , 6.3424 , 1.0000),
# (3.3931 , -0.88042 , 1.0000),
# (-1.3243 , -5.9047 , 1.0000),
# (57.266 , -432.31 , 1.0000),
# (0.78499 , 1.9809 , 1.0000),
# (5.8310 , 10.998 , 1.0000),
# (-4.3906 , 1.2695 , 1.0000),
# (4.7773 , -6.9711 , 1.0000),
# (0.49647 , -2.8599 , 1.0000),
# (190.42 , 2627.0 , 1.0000),
# (-0.66276 , 5.1296 , 1.0000),
# (3.5678 , 2.0810 , 1.0000),
# (-3.2786 , -5.7633 , 1.0000),
# (12.147 , -40.172 , 1.0000),
# (1.0779 , -0.079672 , 1.0000),
# (13.252 , 46.167 , 1.0000),
# (-3.1281 , 5.9532 , 1.0000),
# (3.5180 , -1.8033 , 1.0000),
# (-0.79681 , -5.3145 , 1.0000),
# (137.45 , -1610.7 , 1.0000),
# (0.56738 , 2.6654 , 1.0000),
# (4.9652 , 7.6789 , 1.0000),
# (-4.4162 , -0.72828 , 1.0000),
# (5.5624 , -9.9550 , 1.0000),
# (0.73180 , -2.1641 , 1.0000),
# (70.218 , 587.45 , 1.0000),
# (-1.1745 , 5.7595 , 1.0000),
# (3.4187 , 1.1214 , 1.0000),
# (-2.7098 , -6.2816 , 1.0000),
# (17.106 , -68.902 , 1.0000),
# ]

# class B9EllipticCurveGroupLaw(Scene):
#     def construct(self):
#         num_points_animated = 100
#         grid = NumberPlane(
#             x_range=[-10, 10, 1],
#             y_range=[-10, 10, 1],
#             x_length=6,
#             y_length=6,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 1.5,
#             },
#             axis_config={"numbers_to_include": range(-10, 11, 5)},
#         )
#         self.add(grid)
#         label = MathTex(r'y^2 = x^3 -16x + 16').next_to(grid, UP)
#         gen_label = MathTex(r'(0, 4) \in E(\mathbb Q)').next_to(grid, DOWN).shift(LEFT * 2)
#         bouncing_labels = []
#         for i in range(num_points_animated):
#             pt_lbl = MathTex(rf'{i + 1} \cdot (0, 4) = {np.round(EllipticCurve1616Points[i][0], 1), {np.round(EllipticCurve1616Points[i][1], 1)}}')
#             pt_lbl.next_to(grid, DOWN).shift(RIGHT * 3)
#             bouncing_labels.append(pt_lbl)

#         ec = []
#         for pt in EllipticCurve1616Points:
#             if -10 <= pt[0] <= 10 and -10 <= pt[1] <= 10:
#                 ec.append(Dot(grid.c2p(pt[0], pt[1]), radius=0.03, color=YELLOW))
#                 ec.append(Dot(grid.c2p(pt[0], -pt[1]), radius=0.03, color=YELLOW))
#         ec = VGroup(*ec)

#         points = [
#             Dot(grid.c2p(pt[0], pt[1]))
#             for pt in EllipticCurve1616Points
#         ]
#         points[0].set_color(ORANGE)
#         construction = []
#         for i in range(1, num_points_animated):
#             prev_pt = EllipticCurve1616Points[i - 1]
#             pt = EllipticCurve1616Points[i]
#             construction.append(
#                 VGroup(
#                     Line(start=grid.c2p(prev_pt[0], prev_pt[1]), end=grid.c2p(pt[0], -pt[1]), color=GREEN_B),
#                     Dot(grid.c2p(pt[0], -pt[1]), color=GREEN_D),
#                     Line(start=grid.c2p(pt[0], -pt[1]), end=grid.c2p(pt[0], pt[1]), color=RED_D),
#                 )
#             )
#         self.play(Create(ec), Write(label))
#         self.play(FadeIn(points[0]), Write(gen_label))
#         self.play(Write(bouncing_labels[0]))
#         for i in range(1, min(9, num_points_animated)):
#             self.play(Create(construction[i-1]), run_time=1)
#             self.play(Create(points[i]), TransformMatchingTex(bouncing_labels[i - 1], bouncing_labels[i]))
#             self.play(FadeOut(construction[i-1]), run_time=0.5)
#             if i != 1:
#                 self.play(FadeOut(points[i - 1], run_time=0.5))
#         for i in range(min(9, num_points_animated), num_points_animated):
#             self.play(Create(construction[i-1]), run_time=0.5)
#             self.play(Create(points[i]), TransformMatchingTex(bouncing_labels[i - 1], bouncing_labels[i]), run_time=0.5)
#             self.play(FadeOut(construction[i-1], points[i - 1]), run_time=0.5)

# class B11EllipticCurvesOverFpQR(Scene):
#     def construct(self):
#         ZpAxes = NumberPlane(
#             x_range=[0, 5, 1],
#             y_range=[0, 5, 1],
#             x_length=4,
#             y_length=4,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 2.5,
#             },
#             axis_config={"include_numbers": True},
#         ).to_edge(LEFT)
#         QAxes = Axes(
#             x_range=[-5, 5],
#             y_range=[-5, 5],
#             x_length=4,
#             y_length=4,
#             tips=True,
#             axis_config={
#                 "include_numbers": False,
#                 "tip_width": 0.15,
#                 "tip_height": 0.15
#             },
#         ).next_to(ZpAxes, RIGHT)
#         RAxes = QAxes.copy().next_to(QAxes, RIGHT)
#         zp_label = MathTex(r'E(\mathbb Z / p \mathbb Z)').next_to(ZpAxes, UP)
#         q_label = MathTex(r'E(\mathbb Q)').next_to(QAxes, UP)
#         r_label = MathTex(r'E(\mathbb R)').next_to(RAxes, UP)
#         zp_dots = []
#         for x in range(5):
#             for y in range(5):
#                 if pow(y, 2, 5) == (pow(x, 3, 5) - x + 1) % 5:
#                     zp_dots.append(Dot(ZpAxes.c2p(x, y), color=YELLOW))
#         zp_dots = VGroup(*zp_dots)
#         x_window = [-2, 5]
#         y_window = [-5, 5]
#         max_denom = 150
#         denom_max_lbl = Tex(f'Maximum Denominator: {max_denom}').next_to(QAxes, DOWN).scale(0.7)
#         qpts = []
#         for d1 in range(1, max_denom):
#             for n1 in range((x_window[1] - x_window[0]) * d1 + 1):
#                 for n2 in range((y_window[1] - y_window[0]) * d1 + 1):
#                     x = (n1/d1) + x_window[0]
#                     y = (n2/d1) + y_window[0]
#                     if y ** 2 - x**3 + x - 1 == 0:
#                         if (x, y) not in qpts:
#                             qpts.append((x, y, d1))
#         q_dots = []
#         for qpt in qpts:
#             q_dots.append(Dot(QAxes.c2p(qpt[0], qpt[1]), radius=0.05, color=YELLOW))
#         q_dots = VGroup(*q_dots)
#         r_graph = RAxes.plot_implicit_curve(lambda x, y: y**2 - x**3 - 1 + x, color=YELLOW)

#         self.add(ZpAxes, QAxes, RAxes, zp_label, q_label, r_label)
#         self.play(Create(zp_dots), run_time=1)
#         self.wait()
#         self.play(Write(denom_max_lbl))
#         self.play(Create(q_dots))
#         self.wait()
#         self.play(Create(r_graph))
#         self.wait(6)

# class B11EllipticCurvesOverC(ThreeDScene):
#     def construct(self):
#         curve = Torus(checkerboard_colors=[YELLOW, ORANGE]).scale(0.5)
#         label = MathTex(r'E(\mathbb C)')
#         self.add_fixed_in_frame_mobjects(label)
#         label.shift(UP * 2.5)
#         self.begin_ambient_camera_rotation(0.15)
#         self.set_camera_orientation(phi=PI/3)
#         self.wait(8)
#         self.play(DrawBorderThenFill(curve))
#         self.wait(5)

# def animate_dot_layer(scene, prime, exp, dots, curve_dots, empty_set_label=None, display_all_dots=True, run_time=2):
#     if display_all_dots:
#         anims = []
#         for is_curve_dot, dot in dots[exp]:
#             if is_curve_dot:
#                 anims.append(Create(dot, run_time=0.1))
#             else:
#                 anims.append(Succession(Create(dot), FadeOut(dot), run_time=0.2))
#         scene.play(LaggedStart(*anims, run_time=run_time, lag_ratio=prime/(len(dots[exp]))))
#     else:
#         scene.play(Create(curve_dots[exp]), run_time=run_time)
#     if len(curve_dots[exp]) == 0 and empty_set_label is not None:
#         scene.play(Write(empty_set_label))
#         return False

#     return True

# class AppendixTotalHensel(ThreeDScene):
#     def construct(self):
#         self.next_section(skip_animations=False)
#         self.set_camera_orientation(phi=PI / 5, theta=-PI)
#         self.begin_ambient_camera_rotation(0.1)

#         ZpAxes = Axes(
#             x_range=[-5, 5],
#             y_range=[-5, 5],
#             x_length=6.5,
#             y_length=6.5,
#             tips=True,
#             axis_config={
#                 "include_numbers": False,
#                 "tip_width": 0.15,
#                 "tip_height": 0.15
#             },
#         ).shift(OUT * 2)
#         ZpAxes.save_state()
#         ZmodAxes = []
#         dots = []
#         curve_dots = []
#         max_exp = 3
#         def curve_eq(x, y, q):
#             return (3 * pow(x, 3, q) + 4 * pow(y, 3, q) + 5) % q
#         def is_singular(x, y, q):
#             if (9 * pow(x, 2, q)) % q == 0 and (12 * pow(y, 2, q)) % q == 0:
#                 return True
#             return False
#         for i in range(1, max_exp + 1):
#             q = 5**i
#             size_multiplier = ((2 * (1 / i) + max_exp - i) / (1 + max_exp))
#             grid = NumberPlane(
#                     x_range=[0, q, 1],
#                     y_range=[0, q, 1],
#                     x_length=6.5,
#                     y_length=6.5,
#                     tips=False,
#                     background_line_style={
#                         "stroke_width": 2.5 * size_multiplier,
#                     },
#                     axis_config={"include_numbers": False},
#                 ).shift(IN * (max_exp - i))

#             grid.get_axes().set_opacity(0)
#             grid.save_state()
#             singular_color = ManimColor((235, 52, 85))
#             smooth_color = ManimColor((52, 235, 85))
#             dot_layer = []
#             curve_dot_layer = []
#             for x in range(q):
#                 for y in range(q):
#                     if curve_eq(x, y, q) == 0:
#                         dot_color = singular_color if is_singular(x, y, q) else smooth_color
#                         curve_dot = Dot(point=grid.c2p(x, y), color=dot_color, radius=0.12 * size_multiplier)
#                         dot_layer.append([True, curve_dot])
#                         curve_dot_layer.append(curve_dot)
#                     else:
#                         dot_layer.append(
#                             [False, Dot(point=grid.c2p(x, y), color=BLUE_A, radius=0.08 * size_multiplier)])
#             dots.append(dot_layer)
#             ZmodAxes.append(grid)
#             vgp_curve_layer = VGroup(*curve_dot_layer)
#             vgp_curve_layer.save_state()
#             curve_dots.append(vgp_curve_layer)
#         ZpArrowZmodp3 = Arrow3D(start=ZpAxes.get_center(), end=ZmodAxes[2].get_center(), resolution=6)\
#             .scale(0.8).flip(RIGHT)
#         Zmodp3ArrowZmodp2 = Arrow3D(start=ZmodAxes[2].get_center(), end=ZmodAxes[2].get_center() + IN, resolution=6)\
#             .scale(0.8).flip(RIGHT)
#         Zmodp2ArrowZmodp = Arrow3D(start=ZmodAxes[1].get_center(), end=ZmodAxes[1].get_center() + IN, resolution=6)\
#             .scale(0.8).flip(RIGHT)
#         ZmodpArrows = [
#             Zmodp2ArrowZmodp,
#             Zmodp3ArrowZmodp2,
#             ZpArrowZmodp3
#         ]
#         # p3_to_p2little_arrows = []
#         # for i in range(len(curve_dots[1])):
#         #     for j in range(5):
#         #         index_of_point_above = 5 * i + j
#         #         proj_arrow = Arrow3D(start=curve_dots[2][index_of_point_above].get_center(), end=curve_dots[1][i].get_center(),
#         #                              resolution=5, stroke_width=3).scale(0.8)
#         #         p3_to_p2little_arrows.append(proj_arrow)
#         # p3_to_p2little_arrows = VGroup(*p3_to_p2little_arrows)
#         # p2_to_p1little_arrows = []
#         # for i in range(len(curve_dots[0]) - 1):
#         #     for j in range(5):
#         #         index_of_point_above = 5 * i + j
#         #         proj_arrow = Arrow3D(start=curve_dots[2][index_of_point_above].get_center(), end=curve_dots[1][i + 1].get_center(),
#         #                              resolution=5, stroke_width=3).scale(0.8)
#         #         p2_to_p1little_arrows.append(proj_arrow)
#         # p2_to_p1little_arrows = VGroup(*p2_to_p1little_arrows)

#         ZpLabel = MathTex(r'\mathbb Z_p').next_to(ZpAxes, RIGHT)
#         ZpLabel.save_state()
#         ZmodLabels = [
#             MathTex(r'\mathbb Z / p \mathbb Z').next_to(ZmodAxes[0], RIGHT),
#             MathTex(r'\mathbb Z / p^2\mathbb Z').next_to(ZmodAxes[1], RIGHT),
#             MathTex(r'\mathbb Z / p^3\mathbb Z').next_to(ZmodAxes[2], RIGHT)
#         ]
#         fade_factor = 0.75
#         for label in ZmodLabels:
#             label.save_state()
#         for arrow in ZmodpArrows:
#             arrow.save_state()
#             arrow.flip(RIGHT)
#         graphZp = ZpAxes.plot_implicit_curve(lambda x, y: 3 * (x ** 3) + 4 * (y ** 3) + 5).set_color_by_gradient([PINK, BLUE, PURPLE, GREEN_B, PINK])
#         self.play(Create(ZpAxes), Write(ZpLabel))
#         self.play(Create(graphZp))
#         self.wait(5)
#         self.play(Create(ZmodAxes[2]), Write(ZmodLabels[2]))
#         self.play(Create(curve_dots[2]), Create(ZmodpArrows[2]), VGroup(ZpAxes, ZpLabel).animate.fade(fade_factor), FadeOut(graphZp))
#         self.wait(3)
#         self.play(ZmodpArrows[2].animate.fade(fade_factor))
#         self.play(Create(ZmodAxes[1]), Write(ZmodLabels[1]))
#         self.play(Create(curve_dots[1]), Create(ZmodpArrows[1]), VGroup(ZmodAxes[2], ZmodLabels[2]).animate.fade(fade_factor), FadeOut(curve_dots[2]))
#         self.wait(3)
#         self.play(ZmodpArrows[1].animate.fade(fade_factor))
#         self.play(Create(ZmodAxes[0]), Write(ZmodLabels[0]))
#         self.play(Create(ZmodpArrows[0]), Create(curve_dots[0][1:]), VGroup(ZmodAxes[1], ZmodLabels[1]).animate.fade(fade_factor), FadeOut(curve_dots[1]))
#         self.wait(3)
#         self.play(ZmodpArrows[0].animate.fade(fade_factor))
#         self.wait(2)
#         self.play(FadeOut(curve_dots[0][1:]))

#         #B13Stuff
#         self.next_section(skip_animations=False)
#         self.wait(2)
#         for i in range(max_exp):
#             animate_dot_layer(self, 5, i, dots, curve_dots, run_time=2 + i, display_all_dots=True)
#             self.play(ZmodpArrows[i].animate.restore())
#             self.play(VGroup(ZmodpArrows[i], ZmodAxes[i], curve_dots[i], ZmodLabels[i]).animate.fade(fade_factor))
#             if i < max_exp - 1:
#                 self.play(ZmodLabels[i + 1].animate.restore(), ZmodAxes[i + 1].animate.restore(), ZmodpArrows[i].animate.fade(fade_factor))
#         self.play(ZmodpArrows[max_exp - 1].animate.fade(fade_factor), ZpAxes.animate.restore(), ZpLabel.animate.restore())
#         self.play(Create(graphZp))
#         self.move_camera(phi=0.8 * PI / 2)

#         hensel_strings = []
#         for i in range(max_exp - 1):
#             size_multiplier = ((2 * (1 / (i + 1)) + max_exp - (i + 1)) / (1 + max_exp))
#             hensel_layer = []
#             for j in range(len(curve_dots[i])):
#                 if curve_dots[i][j].color != ManimColor((235, 52, 85)):
#                     index_of_point_above = 5 * j - 1 if j != 0 else 2
#                     hensel_layer.append(
#                         Line3D(
#                             start=curve_dots[i][j].get_center(),
#                             end=curve_dots[i + 1][index_of_point_above].get_center(),
#                             color=ManimColor((52, 235, 85)),
#                             thickness=0.02 * size_multiplier,
#                             resolution=6
#                         )
#                     )
#             hensel_strings.append(VGroup(*hensel_layer))
#         last_strings = []
#         pieces_graph = graphZp.get_pieces(len(curve_dots[max_exp - 1]))
#         for j in range(len(curve_dots[max_exp - 1])):
#             last_strings.append(
#                 Line3D(
#                     start=curve_dots[max_exp - 1][j].get_center(),
#                     end=pieces_graph[len(curve_dots[max_exp - 1]) - j - 1].get_center(),
#                     color=ManimColor((52, 235, 85)),
#                     thickness=0.04 * ((2 * (1 / max_exp)) / (1 + max_exp)),
#                     resolution=6
#                 )
#             )
#         last_strings = VGroup(*last_strings)
#         self.play(ZmodAxes[0].animate.restore(), curve_dots[0].animate.restore(), ZmodLabels[0].animate.restore())
#         self.play(ZmodAxes[1].animate.restore(), curve_dots[1].animate.restore(), ZmodLabels[1].animate.restore())
#         self.play(Create(hensel_strings[0]))
#         self.wait(0.5)
#         self.play(ZmodAxes[2].animate.restore(), curve_dots[2].animate.restore(), ZmodLabels[2].animate.restore())
#         self.play(Create(hensel_strings[1]))
#         self.wait(0.5)
#         self.play(Create(last_strings))
#         self.wait(10)

# class AppendixTwoDimHenselLift(Scene):
#     def construct(self):
#         self.next_section(skip_animations=False)
#         tex_scale_factor = 0.8
#         legend = VGroup(
#             VGroup(Dot(color=ManimColor((52, 235, 85)), radius=0.1), MathTex(r'\nabla f \ne 0')).arrange(RIGHT),
#             VGroup(Dot(color=ManimColor((235, 52, 85)), radius=0.1), MathTex(r'\nabla f = 0')).arrange(RIGHT)
#         ).arrange(DOWN).scale(tex_scale_factor).to_corner(DOWN + LEFT, buff=0.5)
#         self.add(legend)

#         #Bad Primes;
#         # [ <2, 36>, <3, 54>, <5, 12> ]
#         selmer_equation_label = MathTex(r'f(x, y) = 3x^3 + 4y^3 + 5').scale(tex_scale_factor).to_corner(UP + LEFT, buff=0.5).shift(LEFT * 0.1 + DOWN)
#         selmer_gradient_label = MathTex(r'\nabla f = (9x^2, 12y^2)').scale(tex_scale_factor).next_to(legend, UP).shift(RIGHT * 0.3)
#         self.add(selmer_equation_label, selmer_gradient_label)

#         def selmer_eq(x, y, q):
#             return (3 + 4 * pow(x, 3, q) + 5 * pow(y, 3, q)) % q
#         def selmer_is_singular(x, y, q):
#             if (12 * pow(x, 2, q)) % q == 0 and (15 * pow(y, 2, q)) % q == 0:
#                 return True
#             return False

#         self.play(Write(selmer_equation_label), Write(selmer_gradient_label))
#         hasse = Tex(r'Genus: 1\\'
#                     r'Singularities: 0\\'
#                     r'Hasse Bound says for all $p\ge2$\\'
#                     r'$\#C(\mathbb Z / p \mathbb Z) \ge 1$').shift(UP * 2 + RIGHT).scale(tex_scale_factor)
#         self.bad_prime_comp(hasse, tex_scale_factor, selmer_equation_label, MathTex(r'2^{36} \cdot 3^{54} \cdot 5^{12}', color=PINK), r'$p =$ 2, 3, or 5')

#         self.play(Write(legend))
#         # self.find_point_mod_q(2, 1, selmer_equation_label, selmer_eq, selmer_is_singular)
#         self.find_point_mod_q(3, 2, selmer_equation_label, selmer_eq, selmer_is_singular)
#         # self.find_point_mod_q(5, 1, selmer_equation_label, selmer_eq, selmer_is_singular)
#         self.wait()
#         self.play(FadeOut(selmer_equation_label, selmer_gradient_label, legend))

#         self.next_section(skip_animations=True)
#         selmer_like_label = MathTex(r'f(x, y) = 3y^3 + 4x^5 - y^5').scale(tex_scale_factor).to_corner(UP + LEFT,
#                                                                                                         buff=0.5).shift(
#             LEFT * 0.1 + DOWN)
#         selmer_like_gradient_label = MathTex(r'\nabla f = (20x^4, 9y^2 -5y^4)').scale(tex_scale_factor).next_to(legend, UP).shift(
#             RIGHT * 0.3)

#         def selmer_like_eq(x, y, q):
#             return (3 * pow(y, 3, q) + 4 * pow(x, 5, q) - pow(y, 5, q)) % q
#         def selmer_like_is_singular(x, y, q):
#             if (20 * pow(x, 4, q)) % q == 0 and (9 * pow(y, 2, q) - 5 * pow(y, 4, q)) % q == 0:
#                 return True
#             return False

#         self.play(Write(selmer_like_label), Write(selmer_like_gradient_label))
#         hasse = Tex(r'Genus: 2\\'
#                     r'Singularities: 1\\'
#                     r'Hasse Bound says for all $p>16$\\'
#                     r'$\#C(\mathbb Z / p \mathbb Z) \ge 2$').shift(UP * 2 + RIGHT).scale(tex_scale_factor)
#         self.bad_prime_comp(hasse, tex_scale_factor, selmer_like_label,
#                             MathTex(r'2^{160} \cdot 3^{80} \cdot 5^{50}', color=PINK), r'$p \le 16$')

#         self.play(Write(legend))
#         self.find_point_mod_q(2, 3, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.find_point_mod_q(3, 1, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.find_point_mod_q(5, 1, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.find_point_mod_q(7, 1, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.find_point_mod_q(11, 1, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.find_point_mod_q(13, 1, selmer_like_label, selmer_like_eq, selmer_like_is_singular)
#         self.wait()
#         self.play(FadeOut(selmer_like_label, selmer_like_gradient_label, legend))

#         #Potential bad primes
#         # [ <5, 16800>, <31, 480>, <577, 500>, <1601, 500> ]
#         self.next_section(skip_animations=True)

#         spef_equation_label = MathTex(r'x^{125} + 5x^{25} + 25y^5 - 33').scale(tex_scale_factor).to_corner(UP + LEFT,
#                                                                                                         buff=0.5).shift(
#             LEFT * 0.1 + DOWN)
#         spef_gradient_label = MathTex(r'(125x^{124} + 125x^{24}, 125y^4)').scale(tex_scale_factor).next_to(legend, UP).shift(
#             RIGHT)

#         self.play(Write(spef_equation_label), Write(spef_gradient_label))
#         hasse = Tex(r'Genus: 246\\'
#                     r'Singularities: 0\\'
#                     r'Hasse Bound says for all $p\ge242063$\\'
#                     r'$\#C(\mathbb Z / p \mathbb Z) \ge 1$').shift(UP * 2 + RIGHT).scale(tex_scale_factor)
#         self.bad_prime_comp(hasse, tex_scale_factor, spef_equation_label,
#                             MathTex(r'5^{16800} \cdot 31^{480} \cdot 577^{500} \cdot 1601^{500}', color=PINK).scale(0.6), r'$p <$242063')

#         def specific_eq(x, y, q):
#             return (pow(x, 125, q) + 5 * pow(x, 25, q) + 25 * pow(y, 5, q) - 33) % q
#         def specific_eq_sing(x, y, q):
#             if (125 * pow(x, 124, q) + 125 * pow(x, 24, q)) % q == 0 and (125 * pow(y, 4, q)) % q == 0:
#                 return True
#             return False
#         self.play(Write(legend))
#         # self.find_point_mod_q(5, 4, spef_equation_label, specific_eq, specific_eq_sing, display_all_dots=True)

#     def bad_prime_comp(self, info, tex_scale_factor, equation_label, bad_prime_label, p_equal_string):

#         bad_prime_box = VGroup(
#             Rectangle(width=4, height=2),
#             MathTex(r'\textrm{Bad}\_\textrm{Primes}', color=PINK).shift(UP * 0.5),
#             MathTex(r'\dots').shift(DOWN * 0.5),
#         ).shift(DOWN + RIGHT)

#         def process_input(bad_prime_box, equation, solution, num_waits=1):
#             equation.next_to(bad_prime_box, LEFT, buff=1)
#             solution.next_to(bad_prime_box, RIGHT, buff=0.4)
#             solution.save_state()
#             solution.move_to(bad_prime_box.get_center()).scale(0.01).fade(1)
#             self.play(FadeIn(equation), run_time=0.5)
#             self.wait()
#             self.play(equation.animate.move_to(bad_prime_box.get_center()).fade(1).scale(0.01))
#             for _ in range(num_waits):
#                 self.play(ApplyWave(bad_prime_box[2]), run_time=0.5)
#             self.play(Wiggle(bad_prime_box), solution.animate.restore(), run_time=0.5)
#             return solution

#         self.play(Write(info))
#         self.wait()
#         self.play(Create(bad_prime_box))
#         solution = process_input(bad_prime_box, equation_label.copy(), bad_prime_label)
#         conclusion = Tex(r'$C(\mathbb Q_p) \ne \emptyset$\\'
#                          rf'for every prime except possibly {p_equal_string}.', color=GREEN_C).shift(DOWN * 3 + RIGHT)
#         self.wait(2)
#         self.play(Write(conclusion))
#         self.wait(2)
#         self.play(FadeOut(info, bad_prime_box, solution, conclusion))

#     def find_point_mod_q(self, prime, max_exp, equation_label, curve_eq, is_singular, display_all_dots = True, tex_scale_factor = 0.8):
#         prime_label = MathTex(r'p = ', f'{prime}', color=YELLOW).to_corner(UP + RIGHT, buff=0.5)
#         dots = []
#         curve_dots = []
#         grids = []
#         axes_labels = []
#         ring_labels = [MathTex(r'\frac{\mathbb Z}{' + str(prime) + ' \mathbb Z}').next_to(equation_label, DOWN).shift(DOWN)]
#         for i in range(1, max_exp + 1):
#             q = prime ** i
#             size_multiplier = ((2 * (1 / i) + max_exp - i) / (1 + max_exp))
#             label = MathTex(r'\frac{\mathbb Z}{' + f'{prime}^{i}' +' \mathbb Z}').next_to(equation_label, DOWN).shift(DOWN)
#             grid = NumberPlane(
#                 x_range=[0, q, 1],
#                 y_range=[0, q, 1],
#                 x_length=6.5,
#                 y_length=6.5,
#                 tips=False,
#                 background_line_style={
#                     "stroke_width": 2.5 * size_multiplier,
#                 },
#                 axis_config={"include_numbers": False},

#             ).shift(RIGHT)
#             axes_label = grid.get_axis_labels(Tex(r'x').scale(tex_scale_factor), Tex(r'y').scale(tex_scale_factor))
#             singular_color = ManimColor((235, 52, 85))
#             smooth_color = ManimColor((52, 235, 85))
#             dot_layer = []
#             curve_dot_layer = []
#             for x in range(q):
#                 for y in range(q):
#                     if curve_eq(x, y, q) == 0:
#                         dot_color = singular_color if is_singular(x, y, q) else smooth_color
#                         curve_dot = Dot(point=grid.c2p(x, y), color=dot_color, radius=0.15 * size_multiplier)
#                         dot_layer.append([True, curve_dot])
#                         curve_dot_layer.append(curve_dot)
#                     elif display_all_dots:
#                         dot_layer.append([False, Dot(point=grid.c2p(x, y), color=BLUE_A, radius=0.08 * size_multiplier)])
#             dots.append(dot_layer)
#             curve_dots.append(VGroup(*curve_dot_layer))
#             grids.append(grid)
#             axes_labels.append(VGroup(*axes_label))
#             if i > 1:
#                 ring_labels.append(label)

#         self.play(Create(grids[0]), Write(ring_labels[0]), Write(prime_label), Write(axes_labels[0]))

#         empty_set_label = MathTex(r'C(\mathbb Q_p) = \emptyset', color=RED).scale(2).move_to(grids[0])
#         nonempty_set_label = MathTex(r'C(\mathbb Q_p) \ne \emptyset', color=GREEN_B).scale(2).move_to(grids[0])

#         if animate_dot_layer(self, prime, 0, dots, curve_dots, empty_set_label=empty_set_label, display_all_dots=display_all_dots):
#             if max_exp == 1:
#                 self.wait()
#                 self.play(Write(nonempty_set_label))
#                 self.wait()
#                 self.play(FadeOut(nonempty_set_label, curve_dots[0]))
#             else:
#                 self.play(FadeOut(curve_dots[0]))
#             for i in range(1, max_exp):
#                 self.play(ReplacementTransform(grids[i - 1], grids[i]),
#                           TransformMatchingTex(ring_labels[i - 1], ring_labels[i]),
#                           TransformMatchingTex(axes_labels[i - 1], axes_labels[i]))
#                 display_all_dots = display_all_dots if i <3 else False
#                 if animate_dot_layer(self, prime, i, dots, curve_dots, empty_set_label=empty_set_label, display_all_dots=display_all_dots, run_time=2 + i):
#                     if i == max_exp - 1:
#                         self.wait()
#                         self.play(Write(nonempty_set_label))
#                         self.wait()
#                         self.play(FadeOut(nonempty_set_label, curve_dots[i]))
#                     else:
#                         self.wait(2)
#                         self.play(FadeOut(curve_dots[i]))
#                 else:
#                     self.play(FadeOut(empty_set_label))
#                     self.play(FadeOut(grids[i], ring_labels[i], prime_label, curve_dots[i], axes_labels[i]))
#                     return
#         else:
#             self.play(FadeOut(empty_set_label))
#             self.play(FadeOut(grids[0], ring_labels[0], prime_label, curve_dots[0], axes_labels[0]))
#             return
#         self.play(FadeOut(grids[max_exp - 1], ring_labels[max_exp - 1], prime_label, axes_labels[max_exp - 1]))

# class B20TwistExample(Scene):
#     def construct(self):
#         original_grid = NumberPlane(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=4,
#             y_length=4,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 2.5,
#             },
#             axis_config={"include_numbers": True},
#         ).to_edge(LEFT)
#         original_eq = MathTex(r'3x^3', r' + 4y^3 - 5', r' = 0').next_to(original_grid, UP).scale(0.7)
#         y_choice = MathTex(r'y = 1').next_to(original_grid, DOWN).shift(LEFT).scale(0.7)
#         solved_eq = MathTex(r'3x^3', r' - 1', r' = 0').next_to(original_grid, DOWN).shift(DOWN * 0.6).scale(0.7)
#         root = MathTex(r'x = \sqrt[3]{\frac{1}{3}}').next_to(y_choice, RIGHT).scale(0.7)
#         y_choice.shift(DOWN * 0.05)
#         distinguished_point = Dot(point=original_grid.c2p(1/(np.cbrt(3)), 1), color=PINK)
#         selmer_graph = original_grid.plot_implicit_curve(lambda x, y: 3 * x**3 + 4 * y **3 - 5, color=YELLOW)
#         e_prime_grid = Axes(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=2.5,
#             y_length=2.5,
#             tips=False,
#             axis_config={"include_numbers": False},
#         ).next_to(original_grid, RIGHT).shift(RIGHT * 2 + UP * 2)
#         e_prime_label = MathTex(r'E^\prime: y^2 = x^3 - \frac{4821\sqrt[3]{3}}{376}').next_to(e_prime_grid, UP, buff=0).shift(RIGHT *0.3 + DOWN * 0.2).scale(0.7)
#         e_prime_graph = e_prime_grid.plot_implicit_curve(lambda x, y: y**2 - x**3 + (4821 * np.cbrt(3) / (376 )))
#         e_grid = Axes(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=2.5,
#             y_length=2.5,
#             tips=False,
#             axis_config={"include_numbers": False},
#         ).next_to(original_grid, RIGHT).shift(RIGHT * 2 + DOWN * 2)
#         e_label = MathTex(r'E: y^2 = x^3 - 8062156800').next_to(e_grid, DOWN, buff=0.1).scale(0.7)
#         e_graph = e_grid.plot_implicit_curve(lambda x, y: y**2 - x**3 + 5)
#         j_grid = e_grid.copy().center().to_edge(RIGHT, buff=0.2)
#         j_label = MathTex(r'J: y^2 = x^3 - 24300').next_to(j_grid, UP, buff=0.1).scale(0.6)
#         j_graph = j_grid.plot_implicit_curve(lambda x, y: y**2 - x**3 + 3)
#         extension_tower = []
#         extension_tower.append(MathTex(r'\mathbb Q(\sqrt[3]{3})', color=BLUE).next_to(e_prime_grid, RIGHT, buff=0.3)),
#         extension_tower.append(MathTex(r'\mathbb Q', color=BLUE).next_to(e_grid, RIGHT, buff=0.3).scale(0.8)),
#         extension_tower.append(Line(start=extension_tower[0], end=extension_tower[1], color=BLUE).scale(0.8))
#         arrows = []
#         arrows.append(Arrow(start=original_grid.get_edge_center(RIGHT), end=e_prime_grid.get_edge_center(LEFT), stroke_width=3).scale(0.7))
#         arrows.append(MathTex(r'\phi').next_to(arrows[0], UP).scale(0.7))
#         arrows.append(MathTex(r'\phi(P) = \mathcal O').next_to(arrows[1], UP, buff=0.1).scale(0.7))
#         arrows.append(Arrow(start=e_prime_grid.get_edge_center(DOWN), end=e_grid.get_edge_center(UP), stroke_width=3).scale(0.7))
#         arrows.append(MathTex(r'\psi').next_to(arrows[3], LEFT, buff=0.1).shift(RIGHT * 0.7).scale(0.7))
#         arrows.append(CurvedArrow(start_point=extension_tower[1].get_edge_center(RIGHT), end_point=j_grid.get_corner(DOWN), stroke_width=3).scale(0.7))
#         arrows.append(MathTex(r'\chi').next_to(arrows[5], DOWN).shift(RIGHT * 0.4).scale(0.7))

#         self.add(original_grid, e_prime_grid, e_grid)
#         self.play(Create(selmer_graph), Write(original_eq))
#         self.wait()
#         self.play(Write(y_choice))
#         self.play(TransformFromCopy(VGroup(y_choice, original_eq), solved_eq))
#         self.play(TransformFromCopy(solved_eq, root))
#         self.play(Create(distinguished_point))
#         self.play(Create(arrows[0]), Write(arrows[1]))
#         self.play(Write(arrows[2]))
#         self.play(Write(e_prime_label), Create(e_prime_graph))
#         # self.play(Write(extension_tower[0]))
#         self.wait()
#         self.play(Write(arrows[4]), Create(arrows[3]))
#         self.play(Write(e_label), Create(e_graph))
#         # self.play(Create(extension_tower[1]), Write(extension_tower[2]))
#         self.wait()
#         self.play(Create(j_grid))
#         self.play(Create(arrows[5]), Write(arrows[6]))
#         self.play(Write(j_label), Create(j_graph))
#         self.wait()

# hyperell_points = [
# [ 0, np.sqrt(2)],
# [ -0.707107, -0.914213 ],
# [ 3.12132, 10.1569 ],
# [ -3.17157, -9.84062 ],
# [ -1.41024, 1.77053 ],
# [ 1.14424, -2.45004 ],
# [ -0.267628, 1.21238 ],
# [ -0.511183, -1.02270 ],
# [ 1.82457, 4.09044 ],
# [ -1.95935, -3.58044 ],
# [ -1.90759, 3.38032 ],
# [ 1.76513, -3.90355 ],
# [ -0.495888, 1.03378 ],
# [ -0.285557, -1.19814 ],
# [ 1.18114, 2.51168 ],
# [ -1.43710, -1.84150 ],
# [ -3.03182, 8.96819 ],
# [ 2.97676, -9.29906 ],
# [ -0.693473, 0.918871 ],
# [ -0.0209269, -1.39934 ],
# [ 0.735774, 1.94026 ],
# [ -1.13873, -1.18491 ],
# [ -7.80499, 60.8060 ],
# [ 7.79678, -60.9342 ],
# [ -0.878107, 0.915607 ],
# [ 0.285226, -1.60533 ],
# [ 0.370955, 1.66158 ],
# [ -0.926762, -0.940300 ],
# [ 13.2098, 174.580 ],
# [ -13.2127, -174.504 ],
# [ -1.07902, 1.09431 ],
# [ 0.636263, -1.85376 ],
# [ 0.0539035, 1.45183 ],
# [ -0.741313, -0.905192 ],
# [ 3.55308, 12.9800 ],
# [ -3.59214, -12.7010 ],
# [ -1.34673, 1.61120 ],
# [ 1.05474, -2.31238 ],
# [ -0.221339, 1.24889 ],
# [ -0.549149, -0.996315 ],
# [ 1.98851, 4.64893 ],
# [ -2.10475, -4.17314 ],
# [ -1.78895, 2.94350 ],
# [ 1.62618, -3.49936 ],
# [ -0.456454, 1.06325 ],
# [ -0.329957, -1.16273 ],
# [ 1.27885, 2.68932 ],
# [ -1.51008, -2.04445 ],
# [ -2.72926, 7.21300 ],
# [ 2.66106, -7.58064 ],
# [ -0.658651, 0.933220 ],
# [ -0.0728559, -1.36173 ],
# [ 0.810088, 2.01267 ],
# [ -1.18477, -1.26522 ],
# [ -6.03736, 36.3113 ],
# [ 6.02362, -36.4771 ],
# [ -0.843880, 0.905192 ],
# [ 0.225575, -1.56644 ],
# [ 0.434248, 1.70413 ],
# [ -0.962594, -0.966114 ],
# [ 26.2619, 689.727 ],
# [ -26.2626, -689.688 ],
# [ -1.03886, 1.04260 ],
# [ 0.567568, -1.79970 ],
# [ 0.109223, 1.48949 ],
# [ -0.775258, -0.900397 ],
# [ 4.11558, 17.2374 ],
# [ -4.14484, -16.9957 ],
# [ -1.28872, 1.47676 ],
# [ 0.970001, -2.19666 ],
# [ -0.173667, 1.28591 ],
# [ -0.586168, -0.972481 ],
# [ 2.17628, 5.36507 ],
# [ -2.27528, -4.92438 ],
# [ -1.68515, 2.58722 ],
# [ 1.50088, -3.17430 ],
# [ -0.415875, 1.09461 ],
# [ -0.373043, -1.12840 ],
# [ 1.38392, 2.90448 ],
# [ -1.59141, -2.28716 ],
# [ -2.48341, 5.92190 ],
# [ 2.40069, -6.32590 ],
# [ -0.623223, 0.951007 ],
# [ -0.123371, -1.32419 ],
# [ 0.887329, 2.09632 ],
# [ -1.23411, -1.36065 ],
# [ -4.92474, 24.0908 ],
# [ 4.90406, -24.2941 ],
# [ -0.809940, 0.900255 ],
# [ 0.167373, -1.52824 ],
# [ 0.499191, 1.74942 ],
# [ -0.999537, -0.999538 ],
# [ 2163.51, 4.68077E6 ],
# [ -2163.51, -4.68077E6 ],
# [ -1.00046, 1.00046 ],
# [ 0.500809, -1.75058 ],
# [ 0.165961, 1.52731 ],
# [ -0.809108, -0.900199 ],
# [ 4.88171, 24.0766 ],
# [ -4.90258, -23.8723 ],
# [ -1.23537, 1.36321 ],
# [ 0.889268, -2.09855 ],
# [ -0.124595, 1.32327 ],
# [ -0.622344, -0.951485 ],
# [ 2.39485, 6.29949 ],
# [ -2.47795, -5.89460 ],
# [ -1.59352, 2.29370 ],
# [ 1.38661, -2.91032 ],
# [ -0.374086, 1.12757 ],
# [ -0.414863, -1.09540 ],
# [ 1.49795, 3.16714 ],
# [ -1.68276, -2.57932 ],
# [ -2.27983, 4.94530 ],
# [ 2.18124, -5.38512 ],
# [ -0.587067, 0.971930 ],
# [ -0.172478, -1.28683 ],
# [ 0.967972, 2.19405 ],
# [ -1.28736, -1.47373 ],
# [ -4.16061, 17.1271 ],
# [ 4.13158, -17.3679 ],
# [ -0.776090, 0.900335 ],
# [ 0.110600, -1.49042 ],
# [ 0.565905, 1.79844 ],
# [ -1.03790, -1.04145 ],
# [ -26.9160, 724.438 ],
# [ 26.9153, -724.475 ],
# [ -0.963488, 0.966841 ],
# [ 0.435824, -1.70521 ],
# [ 0.224127, 1.56550 ],
# [ -0.843043, -0.905007 ],
# [ 5.99009, 36.0754 ],
# [ -6.00398, -35.9088 ],
# [ -1.18594, 1.26738 ],
# [ 0.811950, -2.01458 ],
# [ -0.0741144, 1.36081 ],
# [ -0.657788, -0.933617 ],
# [ 2.65407, 7.54501 ],
# [ -2.72261, -7.17648 ],
# [ -1.51197, 2.04990 ],
# [ 1.28134, -2.69412 ],
# [ -0.331031, 1.16187 ],
# [ -0.455471, -1.06400 ],
# [ 1.62295, 3.49051 ],
# [ -1.78623, -2.93386 ],
# [ -2.10862, 4.18951 ],
# [ 1.99282, -4.66444 ],
# [ -0.550069, 0.995698 ],
# [ -0.220184, -1.24979 ],
# [ 1.05261, 2.30930 ],
# [ -1.34524, -1.60761 ],
# [ -3.60392, 12.7862 ],
# [ 3.56512, -13.0643 ],
# [ -0.742149, 0.905023 ],
# [ 0.0552460, -1.45276 ],
# [ 0.634550, 1.85236 ],
# [ -1.07801, -1.09291 ],
# [ -13.3760, 178.847 ],
# [ 13.3732, -178.922 ],
# [ -0.927630, 0.940849 ],
# [ 0.372491, -1.66260 ],
# [ 0.283742, 1.60436 ],
# [ -0.877260, -0.915282 ],
# [ 7.74084, 60.0664 ],
# [ -7.74918, -59.9373 ],
# [ -1.13983, 1.18672 ],
# [ 0.737569, -1.94193 ],
# [ -0.0222201, 1.39841 ],
# [ -0.692624, -0.919180 ],
# [ 2.96819, 9.24957 ],
# [ -3.02356, -8.91780 ],
# [ -1.43881, 1.84607 ],
# [ 1.18346, -2.51566 ],
# [ -0.286664, 1.19726 ],
# [ -0.494932, -1.03448 ],
# [ 1.76153, 3.89249 ],
# [ -1.90448, -3.36844 ],
# [ -1.96266, 3.59346 ],
# [ 1.82835, -4.10262 ],
# [ -0.512128, 1.02202 ],
# [ -0.266507, -1.21327 ],
# [ 1.14197, 2.44635 ],
# [ -1.40861, -1.76628 ],
# [ -3.18070, 9.89894 ],
# [ 3.13074, -10.2143 ],
# [ -0.707952, 0.913944 ],
# [ 0.00130776, -1.41514 ],
# [ 0.705338, 1.91264 ],
# [ -1.12025, -1.15518 ],
# [ -8.90097, 79.1276 ],
# [ 8.89467, -79.2400 ],
# [ -0.892620, 0.921740 ],
# [ 0.310721, -1.62196 ],
# [ 0.344838, 1.64433 ],
# [ -0.911975, -0.931543 ],
# [ 10.9276, 119.511 ],
# [ -10.9317, -119.420 ],
# [ -1.09650, 1.11918 ],
# [ 0.665745, -1.87828 ],
# [ 0.0310890, 1.43603 ],
# [ -0.726986, -0.908487 ],
# [ 3.35870, 11.6609 ],
# ]
# ell_points = [
# [ 0.000000, 2.00000 ],
# [ 4.00000, 6.00000 ],
# [ -3.00000, 1.00000 ],
# [ 3.11111, -3.03704 ],
# [ -0.489796, -2.79300 ],
# [ 96.2500, -943.875 ],
# [ 0.325289, 1.19671 ],
# [ 5.77302, 12.2563 ],
# [ -2.61672, 2.64886 ],
# [ 2.67821, -1.33589 ],
# [ -1.12677, -3.40347 ],
# [ 24.1239, -117.687 ],
# [ 0.491022, 0.436131 ],
# [ 9.65274, 28.7433 ],
# [ -1.97686, 3.47697 ],
# [ 2.53506, -0.105977 ],
# [ -1.84494, -3.53266 ],
# [ 10.8379, -34.5011 ],
# [ 0.504859, -0.299685 ],
# [ 20.2441, 90.2142 ],
# [ -1.25617, 3.47378 ],
# [ 2.63265, 1.08872 ],
# [ -2.51284, -2.86980 ],
# [ 6.26857, -14.1483 ],
# [ 0.367597, -1.05304 ],
# [ 68.6124, 567.854 ],
# [ -0.597392, 2.92676 ],
# [ 3.00407, 2.66036 ],
# [ -2.95575, -1.35026 ],
# [ 4.24051, -6.80650 ],
# [ 0.0724016, -1.84964 ],
# [ 2827.04, 150314. ],
# [ -0.0780592, 2.15035 ],
# [ 3.78785, 5.29570 ],
# [ -3.03082, 0.637033 ],
# [ 3.23306, -3.45391 ],
# [ -0.387346, -2.65342 ],
# [ 144.714, -1740.54 ],
# [ 0.277547, 1.34201 ],
# [ 5.34275, 10.6661 ],
# [ -2.71174, 2.39855 ],
# [ 2.73334, -1.59828 ],
# [ -1.00033, -3.31687 ],
# [ 29.2509, -157.472 ],
# [ 0.472037, 0.573483 ],
# [ 8.66071, 24.1731 ],
# [ -2.10615, 3.39214 ],
# [ 2.54305, -0.319073 ],
# [ -1.71145, -3.56071 ],
# [ 12.2683, -41.8612 ],
# [ 0.513576, -0.163879 ],
# [ 17.2388, 70.6334 ],
# [ -1.38787, 3.52557 ],
# [ 2.59615, 0.853737 ],
# [ -2.40121, -3.06019 ],
# [ 6.84214, -16.4188 ],
# [ 0.404531, -0.911017 ],
# [ 51.3784, 367.721 ],
# [ -0.709885, 3.05309 ],
# [ 2.91055, 2.31768 ],
# [ -2.89863, -1.68362 ],
# [ 4.51360, -7.73593 ],
# [ 0.139156, -1.69984 ],
# [ 706.763, 18789.2 ],
# [ -0.161763, 2.29997 ],
# [ 3.60053, 4.67680 ],
# [ -3.04782, 0.265889 ],
# [ 3.37154, -3.91830 ],
# [ -0.290233, -2.50946 ],
# [ 241.702, -3757.42 ],
# [ 0.224309, 1.48890 ],
# [ 4.96756, 9.31892 ],
# [ -2.79682, 2.12067 ],
# [ 2.79868, -1.87924 ],
# [ -0.877418, -3.21619 ],
# [ 36.2196, -217.323 ],
# [ 0.447864, 0.711983 ],
# [ 7.82299, 20.4982 ],
# [ -2.23168, 3.27701 ],
# [ 2.55911, -0.535620 ],
# [ -1.57739, -3.56291 ],
# [ 14.0147, -51.4249 ],
# [ 0.517192, -0.0284298 ],
# [ 14.8649, 56.3004 ],
# [ -1.52114, 3.55660 ],
# [ 2.56831, 0.628178 ],
# [ -2.28300, -3.21943 ],
# [ 7.50977, -19.1689 ],
# [ 0.436152, -0.770552 ],
# [ 39.9150, 251.550 ],
# [ -0.826962, 3.17020 ],
# [ 2.82935, 2.00370 ],
# [ -2.82935, -1.99630 ],
# [ 4.82435, -8.81413 ],
# [ 0.200294, -1.55103 ],
# [ 314.121, 5567.09 ],
# [ -0.251067, 2.44800 ],
# [ 3.43511, 4.12958 ],
# [ -3.05078, -0.108688 ],
# [ 3.52853, -4.43891 ],
# ]
# two_torsion = [
# (-3.05137 , -0.000000 , 1.00000),
# (0.517304 , -0.000000 , 1.00000),
# (2.53407 , -0.000000 , 1.00000) 
# ]
# three_torsion = [
# (3.63495 , 4.79047 , 1.00000),
# ]
# four_torsion = [
# (0.517304 , -0.000000 , 1.00000),
# (2.53407 , -0.000000 , 1.00000),
# (-3.05137 , 0.00756442 , 1.00000),
# (-0.822198, 3.16572 , 1.00000),
# (5.89034 , 12.6984 , 1.00000) 
# ]

# class B21JacobianExample(Scene):
#     def construct(self):
#         self.next_section(skip_animations=False)
#         num_points_animated = 40
#         grid1 = NumberPlane(
#             x_range=[-10, 10, 1],
#             y_range=[-10, 10, 1],
#             x_length=6,
#             y_length=6,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 1.5,
#             },
#             axis_config={"numbers_to_include": range(-10, 11, 5)},
#         ).to_edge(LEFT)
#         grid2 = grid1.copy().to_edge(RIGHT)
#         hyper_ell_eq = MathTex(r'y^2 = x^4 + 2x +2').next_to(grid2, UP)
#         jac_eq = MathTex(r'y^2 = x^3 - 8x + 4').next_to(grid1, UP)
#         fields = [
#             MathTex(r'J(\mathbb Q)').next_to(grid1, DOWN, buff=0.1).scale(0.8),
#             MathTex(r'\mathbb C\left (\mathbb Q\left ( \sqrt{2}, \sqrt{303}\right )\right )').next_to(grid2, DOWN, buff=0).scale(0.8),
#         ]
#         self.add(grid1, grid2, hyper_ell_eq, jac_eq, *fields)
#         curve_rat_points = []
#         jac_rat_points = []
#         for i in range(100):
#             pt = ell_points[i]
#             jac_rat_points.append(Dot(grid1.c2p(pt[0], -pt[1]), color=YELLOW, radius=0.03))
#             jac_rat_points.append(Dot(grid1.c2p(pt[0], pt[1]), color=YELLOW, radius=0.03))
#         for i in range(100):
#             pt = hyperell_points[i]
#             curve_rat_points.append(Dot(grid2.c2p(pt[0], -pt[1]), color=YELLOW, radius=0.03))
#             curve_rat_points.append(Dot(grid2.c2p(pt[0], pt[1]), color=YELLOW, radius=0.03))
#         curve_rat_points = VGroup(*curve_rat_points)
#         jac_rat_points = VGroup(*jac_rat_points)
#         gen_point = Dot(grid1.c2p(*ell_points[0]), color=ORANGE, radius=0.1)
#         first_curve_point = Dot(grid2.c2p(*hyperell_points[0]), color=PINK, radius=0.1)
#         construction = []
#         g1arrow = []
#         highighted_jac_dots = []
#         highlighted_curve_dots = []
#         for i in range(1, num_points_animated):
#             prev_pt = ell_points[i - 1]
#             pt = ell_points[i]
#             construction.append(
#                 VGroup(
#                     Line(start=grid1.c2p(prev_pt[0], prev_pt[1]), end=grid1.c2p(pt[0], -pt[1]), color=GREEN_B),
#                     Dot(grid1.c2p(pt[0], -pt[1]), color=GREEN_D),
#                     Line(start=grid1.c2p(pt[0], -pt[1]), end=grid1.c2p(pt[0], pt[1]), color=RED_D),
#                 )
#             )
#             prev_pt = hyperell_points[i - 1]
#             pt = hyperell_points[i]
#             g1arrow.append(
#                 Line(start=grid2.c2p(prev_pt[0], prev_pt[1]), end=grid2.c2p(pt[0], pt[1]), color=PINK, stroke_width=4)
#             )
#             highighted_jac_dots.append(Dot(grid1.c2p(*ell_points[i]), radius=0.1))
#             highlighted_curve_dots.append(Dot(grid2.c2p(*hyperell_points[i]), color=PINK, radius=0.1))

#         self.play(Create(jac_rat_points), Create(curve_rat_points))
#         self.play(Create(gen_point))
#         self.wait()
#         self.play(Create(first_curve_point))
#         for i in range(min(num_points_animated, 5) - 1):
#             self.play(Create(construction[i]), run_time=0.5)
#             self.play(Create(highighted_jac_dots[i]), run_time=0.5)
#             self.play(Create(g1arrow[i]), run_time=0.5)
#             self.play(Create(highlighted_curve_dots[i]), run_time=0.5)
#             if i == 0:
#                 self.play(FadeOut(construction[i], g1arrow[i], first_curve_point), run_time=0.5)
#             else:
#                 self.play(FadeOut(g1arrow[i], construction[i], highighted_jac_dots[i - 1], highlighted_curve_dots[i - 1]), run_time=0.5)

#         if num_points_animated > 5:
#             for i in range(4, num_points_animated - 1):
#                 self.play(Create(construction[i]), Create(g1arrow[i]), run_time=0.4)
#                 self.play(Create(highighted_jac_dots[i]), Create(highlighted_curve_dots[i]), run_time=0.4)
#                 self.play(
#                         FadeOut(g1arrow[i], construction[i], highighted_jac_dots[i - 1], highlighted_curve_dots[i - 1]),
#                         run_time=0.25)

#         self.play(FadeOut(jac_rat_points, highighted_jac_dots[num_points_animated - 2], highlighted_curve_dots[num_points_animated - 2], *fields))
#         action_label = MathTex(r'P = (0, 2) \curvearrowright C\left (\mathbb Q(\sqrt{2}, \sqrt{303}) \right)').to_edge(DOWN, buff=0.3)
#         self.wait()
#         self.play(Write(action_label))
#         self.wait()
#         for _ in range(5):
#             self.play(gen_point.animate.set_color(PINK).scale(1.5))
#             anims = []
#             for j in range(len(curve_rat_points) - 2):
#                 anims.append(AnimationGroup(curve_rat_points[j].animate.move_to(curve_rat_points[(j + 1) % len(curve_rat_points)]), run_time=2))
#             self.play(AnimationGroup(*anims))
#             self.play(gen_point.animate.set_color(ORANGE).scale(1 / 1.5))

#         self.play(FadeOut(action_label, gen_point))


#         self.next_section(skip_animations=True)
#         rest_pts = []
#         for i in range(num_points_animated, len(hyperell_points)):
#             pt = hyperell_points[i]
#             rest_pts.append(Dot(grid2.c2p(pt[0], -pt[1]), color=YELLOW, radius=0.03))
#             rest_pts.append(Dot(grid2.c2p(pt[0], pt[1]), color=YELLOW, radius=0.03))
#         rest_pts = VGroup(*rest_pts)
#         all_pts = [*curve_rat_points, *rest_pts]
#         torsion_labels = [
#             MathTex(r'E(\Bar{\mathbb Q})[2]').next_to(grid1, DOWN, buff=0.1).scale(0.8),
#             MathTex(r'E(\Bar{\mathbb Q})[3]').next_to(grid1, DOWN, buff=0.1).scale(0.8),
#             MathTex(r'E(\Bar{\mathbb Q})[4]').next_to(grid1, DOWN, buff=0.1).scale(0.8),
#         ]
#         torsion_points = []
#         torsion_points.append(
#             VGroup(*[
#                 Dot(grid1.c2p(pt[0], pt[1]), color=YELLOW)
#                 for pt in two_torsion
#             ])
#         )
#         torsion_points.append(
#             VGroup(*[
#                 Dot(grid1.c2p(pt[0], pt[1]), color=YELLOW)
#                 for pt in three_torsion
#             ])
#         )
#         torsion_points.append(
#             VGroup(*[
#                 Dot(grid1.c2p(pt[0], pt[1]), color=YELLOW)
#                 for pt in four_torsion
#             ])
#         )
#         self.play(Create(rest_pts))
#         self.play(Write(torsion_labels[0]))
#         self.play(Create(torsion_points[0]))
#         for i in range(len(two_torsion)):
#             self.play(torsion_points[0][i].animate.set_color(PINK).scale(1.5))
#             anims = []
#             for j in range(len(all_pts) - 2):
#                 anims.append(AnimationGroup(all_pts[j].animate.move_to(all_pts[(j + 2 + i) % len(all_pts)]), run_time=2))
#             self.play(AnimationGroup(*anims))
#             self.play(torsion_points[0][i].animate.set_color(YELLOW).scale(1/1.5))
#         self.play(FadeOut(torsion_points[0], torsion_labels[0]))
#         self.play(Write(torsion_labels[1]))
#         self.play(Create(torsion_points[1]))
#         for i in range(len(three_torsion)):
#             self.play(torsion_points[1][i].animate.set_color(PINK).scale(1.5))
#             anims = []
#             for j in range(len(all_pts) - 2):
#                 anims.append(AnimationGroup(all_pts[j].animate.move_to(all_pts[(j + i + 6) % len(all_pts)]), run_time=2))
#             self.play(AnimationGroup(*anims))
#             self.play(torsion_points[1][i].animate.set_color(YELLOW).scale(1/1.5))
#         self.play(FadeOut(torsion_points[1], torsion_labels[1]))
#         self.play(Write(torsion_labels[2]))
#         self.play(Create(torsion_points[2]))

#         for i in range(len(four_torsion)):
#             self.play(torsion_points[2][i].animate.set_color(PINK).scale(1.5))
#             anims = []
#             for j in range(len(all_pts) - 2):
#                 anims.append(
#                     AnimationGroup(all_pts[j].animate.move_to(all_pts[(j + (i**4 + 1)) % len(all_pts)]), run_time=2))
#             self.play(AnimationGroup(*anims))
#             self.play(torsion_points[2][i].animate.set_color(YELLOW).scale(1 / 1.5))

# class B22Torsors(Scene):
#     def construct(self):
#         grid1 = NumberPlane(
#             x_range=[-5, 5, 1],
#             y_range=[-5, 5, 1],
#             x_length=3.5,
#             y_length=3.5,
#             tips=False,
#             background_line_style={
#                 "stroke_width": 1.5,
#             },
#             axis_config={"include_numbers": False},
#         ).to_edge(LEFT)
#         jac_eq = MathTex(r'y^2 = x^3 - 8x + 4').next_to(grid1, UP)
#         jac_graph = grid1.plot_implicit_curve(lambda x, y: y**2 - x**3 + 8 * x - 4, color=YELLOW)
#         grid2 = grid1.copy().scale(0.6).to_edge(UP).shift(RIGHT * 5)
#         hyper_ell_eq = MathTex(r'y^2 = x^4 + 2x +2').next_to(grid2, UP, buff=0).scale(0.6)
#         hyper_ell_graph = grid2.plot_implicit_curve(lambda x, y: y**2 - x**4 - 2 * x - 2, color=YELLOW)
#         equiv = MathTex(r'\sim').next_to(grid2, RIGHT)
#         grid3 = grid1.copy().scale(0.6).next_to(equiv, RIGHT)
#         equiv_ex = MathTex(r'\left(y+x\right)^{2}=x^{4}+2x+2').next_to(grid3, UP, buff=0).scale(0.6)
#         equiv_ex_graph = grid3.plot_implicit_curve(lambda x, y: (x + y)**2 - x**4 - 2 * x - 2, color=YELLOW)
#         dots = MathTex(r'\dots').next_to(grid3, RIGHT)
#         grid4 = grid1.copy().scale(0.6).next_to(grid2, DOWN, buff=0.5)
#         new_ex = MathTex(r'y^{2}=\left(x-1\right)^{4}+2\left(x-1\right)+2').next_to(grid4, UP, buff=0).scale(0.5)
#         new_ex_graph = grid4.plot_implicit_curve(lambda x, y: y ** 2 - (x - 1) ** 4 - 2 * (x - 1)-2, color=YELLOW)
#         equiv_copy = equiv.copy().next_to(grid4, RIGHT)
#         grid5 = grid1.copy().scale(0.6).next_to(equiv_copy, RIGHT)
#         equiv_new_ex = MathTex(r'9y^{2}=\left(x-1\right)^{4}+2\left(x-1\right)+2').next_to(grid5, UP, buff=0).scale(0.5).shift(RIGHT * 0.7)
#         equiv_new_ex_graph = grid5.plot_implicit_curve(lambda x, y: (y + 1) ** 2 - (x - 1) ** 4 -2 * (x - 1)-2, color=YELLOW)
#         dots_copy = dots.copy().next_to(grid5, RIGHT)
#         grid6 = grid1.copy().scale(0.6).next_to(grid4, DOWN, buff=0.5)
#         another_new_ex = MathTex(r'y^{2}=\frac{16}{9}x^{4}+\frac{4}{9}x+\frac{2}{9}').next_to(grid6, UP, buff=-0.25).scale(0.5)
#         another_new_ex_graph = grid6.plot_implicit_curve(lambda x, y: y ** 2 - (16/9)* x ** 4 - (4/9) * x - (2/9), color=YELLOW)
#         final_dots_copy = dots.copy().next_to(grid6, RIGHT)
#         arrows = [
#             Arrow(start=grid1.get_edge_center(RIGHT), end=grid2.get_edge_center(LEFT)),
#             Arrow(start=grid1.get_edge_center(RIGHT), end=grid4.get_edge_center(LEFT)),
#             Arrow(start=grid1.get_edge_center(RIGHT), end=grid6.get_edge_center(LEFT))
#         ]
#         fields = [
#             MathTex(r'Q\left ( \sqrt{2}, \sqrt{303}\right )').next_to(arrows[0], UP, buff=0.1).scale(0.6),
#             MathTex(r'\mathbb Q').next_to(arrows[1], UP, buff=0.1).scale(0.6),
#             MathTex(r'Q\left ( \sqrt{2}, \sqrt{3}, \sqrt{303}\right )').next_to(arrows[2], DOWN, buff=0.1).scale(0.5).shift(LEFT * 0.1),
#         ]
#         self.add(grid1, grid2, grid3, grid4, grid5, grid6)
#         self.play(Write(jac_eq), Create(jac_graph))
#         self.play(LaggedStart(*[Create(arrows[i]) for i in range(len(arrows))]))
#         self.play(Write(fields[0]))
#         self.play(Write(hyper_ell_eq), Create(hyper_ell_graph))
#         self.play(Write(equiv))
#         self.play(Write(equiv_ex), Create(equiv_ex_graph))
#         self.play(Write(dots), run_time=0.5)
#         self.wait(0.5)
#         self.play(Write(fields[1]))
#         self.play(Write(new_ex), Create(new_ex_graph))
#         self.play(Write(equiv_copy))
#         self.play(Write(equiv_new_ex), Create(equiv_new_ex_graph))
#         self.play(Write(dots_copy), run_time=0.5)
#         self.wait(0.5)
#         self.play(Write(fields[2]))
#         self.play(Write(another_new_ex), Create(another_new_ex_graph))
#         self.play(Write(final_dots_copy), run_time=0.5)
