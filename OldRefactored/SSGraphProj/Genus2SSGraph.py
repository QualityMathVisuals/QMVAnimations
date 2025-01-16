from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup
from sage.all import *
from avisogenies_sage import *
from sympy import mod_inverse

from SupersingularGraphs.HoveringECGraph import HoveringECGraphScene
from SupersingularGraphs.SSGraph import BreadthFirstSSGraphScene
from sqisignutilities import Cornacchia
from sage.schemes.elliptic_curves.hom_composite import EllipticCurveHom_composite
from sage.schemes.elliptic_curves.weierstrass_morphism import *

from SIDH import SIDHScene


class Genus2SSGraphScene(HoveringECGraphScene):
    def __init__(self):
        super().__init__(p=157, l=15)
        self.vertex_radius = 0.3
        j0 = 0
        self.levels = {0: [j0]}
        self.j_invars = [j0]
        self.vertices = {j0: self.j_vertex(j0)}

    def j_vertex(self, j_inv, vertex_color=BLUE_B):
        dot = Circle(radius=self.vertex_radius, color=vertex_color, fill_opacity=0.4)
        return OpenGLVGroup(dot, self.elliptic_curve_product())

    def l_label(self):
        return MathTex(fr'\ell = {2}')

    def draw_neighbors(self, j_curr, l=None, animate=True, vertex_select_color=GREEN_B, vertex_color=BLUE_B, run_time=1):
        if l is None:
            l = self.l
        # self.ring_spacing_range = [self.vertex_radius * 2 + 0.1 * l, self.vertex_radius * 2 + 0.6 * l]

        current_dot = self.vertices[j_curr][0]
        # Turn current dot green.
        if animate:
            self.play(
                current_dot.animate.set_color(vertex_select_color),
                run_time=0.5 * run_time
            )

        # Calculate unique neighboring j_invariants
        neighbor_js = []
        non_new_js = []
        neighboring_edges = {}
        non_new_edges = {}
        neighboring_vertices = []
        for root, mult in X0j.roots():
            # Create edges to neighbors
            if root not in self.j_invars:
                self.j_invars.append(root)
                neighbor_js.append(root)
            elif root != j_curr:
                non_new_js.append(root)
                # This sometimes draws an overlapping edge (to the previous node)
                non_new_edges[root] = self.j_edge(j_curr, root)

        # Calculate placement of neighbor (with random variation)
        J = len(neighbor_js)
        if J > 0:
            if self.current_level == 0:
                unit_angle = TAU / J
                init_angle = -PI / 2 + (PI / 2) * (l%2)
            elif self.current_level == 1:
                unit_angle = TAU / J
                init_angle = -PI / 2
            elif self.current_level == 2 and l == 2:
                unit_angle = TAU / J
                init_angle = -PI / 2
            else:
                unit_angle = (PI / J) * (self.unit_angle_scaling_range[0] + random() * (
                            self.unit_angle_scaling_range[1] - self.unit_angle_scaling_range[0]))
                init_angle = -PI / 2 + ((self.init_angle_offset_range[1] - self.init_angle_offset_range[0]) * random() + self.init_angle_offset_range[0]) * unit_angle  # Prefer bias so that there is fanning
            for k, j_next in enumerate(neighbor_js):
                dot = self.j_vertex(j_next, vertex_color=vertex_color).move_to(current_dot.get_center(), aligned_edge=IN)
                rot_angle = init_angle + k * unit_angle
                if self.current_level == 0 or (self.current_level == 2 and l == 2):
                    direction_vec = RIGHT * self.ring_spacing_range[0] * (1 + self.current_level * self.ring_spacing_min_scaling)
                else:
                    direction_vec = current_dot.get_center() / np.linalg.norm(current_dot.get_center())
                    scaling_factor = (1 - self.ring_spacing_min_scaling) * random() + self.ring_spacing_min_scaling

                    if self.max_level_length is not None:
                        direction_vec *= self.ring_spacing_range[0] + (min(self.current_level, self.max_level_length) * scaling_factor * (
                                self.ring_spacing_range[1] - self.ring_spacing_range[0]))
                    else:
                        direction_vec *= self.ring_spacing_range[0] + (self.current_level * scaling_factor * (
                                                 self.ring_spacing_range[1] - self.ring_spacing_range[0]))

                direction_vec = rotate_vector(direction_vec, rot_angle, OUT)
                dot.shift(direction_vec)
                self.vertices[j_next] = dot
                neighboring_vertices.append(dot)
                neighboring_edges[j_next] = self.j_edge(j_curr, j_next)

        neighboring_edges = neighboring_edges | non_new_edges
        for j_next, edge in neighboring_edges.items():
            self.edges[(j_curr, j_next)] = edge

        if animate:
            # Draw neighboring edges
            if len(neighboring_edges) > 0:
                self.play(*[
                    Create(edge)
                    for edge in neighboring_edges.values()
                ], run_time=0.5 * run_time)

            # Draw neighboring nodes
            if len(neighboring_vertices) > 0:
                self.play(*[
                    Create(dot)
                    for dot in neighboring_vertices
                ], run_time=run_time)

            # Revert current dot color
            self.play(
                current_dot.animate.set_color(vertex_color),
                run_time=0.5 * run_time
            )

        return neighbor_js, neighboring_vertices, neighboring_edges, non_new_js

    def elliptic_curve_product(self):
        OpenGLVGroup(self.elliptic_curve_mob(randint(0,100000)).center(), MathTex(r'\times'), self.elliptic_curve_mob(randint(0,100000)).center()).arrange(RIGHT).shift(OUT)



class FindingEndomorphismGamma(SIDHScene):
    def __init__(self, e2=63, e3=30):
        super().__init__(e2=e2, e3=e3, max_walk_lengths= [20, 20])
        la, lb = self.PA.order(), self.PB.order()
        print(la, lb)
        bigL, smallL = max(la, lb), min(la, lb)
        if la > lb:
            pass
        else:
            bigP, bigQ = self.PB, self.QB
            smallP, smallQ = self.PA, self.QA
            leaked_torsion_info = self.PhiA(self.PB), self.PhiA(self.QB)
            print(smallP.order(), smallQ.order())
            print(leaked_torsion_info[0].order(), leaked_torsion_info[1].order())

        c = bigL - smallL
        print(c)
        x, y = 1, 1
        u, v = None, None
        while u is None:
            uv = Cornacchia(c, 1)
            if len(uv) > 0:
                u, v = uv[0], uv[1]
            else:
                x += 1
                c = x * bigL - smallL
                print(x, y, c)

        print(c == u**2 + v**2)
        print(f'{u}^2 + {v}^2 = {c} = {x} * {bigL} - {smallL}')
        #Construct endomorphism gamma on E0
        print(r'Constructing gamma')
        if self.init_j_invariant != 1728:
            raise Exception("init_j_invariant must be 1728")

        i = sqrt(self.Fq(-1))
        i_endo = WeierstrassIsomorphism(E=self.E0, urst=(i, 0, 0, 0))
        gamma = self.E0.scalar_multiplication(u) + i_endo * self.E0.scalar_multiplication(v)
        gamma_dual = gamma.dual()

        #Somehow there is a way to calculate this without cheating
        goal_isog = (self.PhiA * gamma_dual)

        #Kernel of the \Phi:E0 × EA → E0 × C is given by
        kern = [
            (smallL * bigP, -goal_isog(bigP)),
            (smallL * bigQ, -goal_isog(bigQ))
        ]
        print(kern)






    def get_neighbors(self, J):
        pass

    def setup(self):
        pass

    def construct(self):
        self.interactive_embed()

def _glue_curves(C, E):
    Fp2 = C.base()
    Rx = PolynomialRing(Fp2, name="x")
    x = Rx.gens()[0]

    P_c, Q_c = C.torsion_basis(2)
    P, Q = E.torsion_basis(2)

    a1, a2, a3 = P_c[0], Q_c[0], (P_c + Q_c)[0]
    b1, b2, b3 = P[0], Q[0], (P + Q)[0]

    # Compute coefficients
    M = sage.all.Matrix(Fp2, [
        [a1 * b1, a1, b1],
        [a2 * b2, a2, b2],
        [a3 * b3, a3, b3]]
    )
    R, S, T = M.inverse() * vector(Fp2, [1, 1, 1])
    RD = R * M.determinant()
    da = (a1 - a2) * (a2 - a3) * (a3 - a1)
    db = (b1 - b2) * (b2 - b3) * (b3 - b1)

    s1, t1 = - da / RD, db / RD
    s2, t2 = -T / R, -S / R

    a1_t = (a1 - s2) / s1
    a2_t = (a2 - s2) / s1
    a3_t = (a3 - s2) / s1
    h = s1 * (x ** 2 - a1_t) * (x ** 2 - a2_t) * (x ** 2 - a3_t)

    H = HyperellipticCurve(h)
    J = H.jacobian()

    return h, J


manim_configuration = {"quality": "high_quality", "preview": True, "disable_caching": True,
                       # "from_animation_number": 0, "upto_animation_number": 70,
                       "renderer": "opengl", "write_to_movie": False, "show_file_in_browser": False, }
if __name__ == '__main__':
    with tempconfig(manim_configuration):
        scene = Genus2SSGraphScene()
        scene.render()
