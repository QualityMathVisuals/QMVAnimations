from cProfile import label

from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup
from sage.all import *
from sage.schemes.elliptic_curves.hom_composite import EllipticCurveHom_composite
from Utilities import choose_random_int, to_base
from SupersingularGraphs.SSGraph import TraverseSSGraphScene


class SIDHScene(TraverseSSGraphScene):
    def __init__(self, e2=216, e3=137, max_walk_lengths=None, init_j_invariant=None, swap_alice_bob= False):
        if max_walk_lengths is None:
            max_walk_lengths = [e2, e3]
        p = (2 ** e2) * (3 ** e3) - 1
        self.e2, self.e3 = e2, e3
        self.p = p
        Fq = FiniteField(p ** 2, 'w')
        self.Fq = Fq
        w = Fq.gen()
        if init_j_invariant is None:
            if p % 4 == 3:
                E0 = EllipticCurve(Fq, [1, 0])
            elif p % 3 == 2:
                E0 = EllipticCurve(Fq, [0, 0.9])
            else:
                raise Exception("p must be congruent to 3 mod 4 or 2 mod 3")
        else:
            E0 = EllipticCurve_from_j(Fq(init_j_invariant))

        self.E0 = E0
        self.init_j_invariant = E0.j_invariant()

        torsion_subgroup_sizes = [2 ** min(e2, max_walk_lengths[0]), 3 ** min(e3, max_walk_lengths[1])]
        self.PA, self.QA = E0.torsion_basis(torsion_subgroup_sizes[0])
        self.PB, self.QB = E0.torsion_basis(torsion_subgroup_sizes[1])
        nA= randrange(0, torsion_subgroup_sizes[0])
        nB= randrange(0, torsion_subgroup_sizes[1])
        self.A = nA * self.PA +  self.QA  # Secret keys
        self.nA = nA
        self.nB = nB
        self.B = nB * self.PB +  self.QB
        self.PhiA = EllipticCurveHom_composite(E0, self.A)
        self.PhiB = EllipticCurveHom_composite(E0, self.B)
        self.EA = self.PhiA.codomain()
        self.EB = self.PhiB.codomain()

        predetermined_path_A = self.path_from_isogeny(self.PhiA)
        predetermined_path_B = self.path_from_isogeny(self.PhiB)

        ls = [2, 3]
        path_specifications = [predetermined_path_A, predetermined_path_B]
        path_lengths = [len(predetermined_path_A) + 1, len(predetermined_path_B) + 1]
        num_paths = 2
        if swap_alice_bob:
            ls= [3, 2]
            path_specifications = [predetermined_path_B, predetermined_path_A]
            path_lengths = [len(predetermined_path_B) + 1, len(predetermined_path_A) + 1]
        self.swap_alice_bob = swap_alice_bob
        super().__init__(
            p, ls=ls,
            path_specifications=path_specifications,
            path_lengths=path_lengths,
            num_paths=num_paths,
            animate_camera_frames=False
        )
        if swap_alice_bob:
            temp = self.vertex_select_color[1]
            self.vertex_select_color[1] = self.vertex_select_color[0]
            self.vertex_select_color[0] = temp
            temp = self.vertex_color[1]
            self.vertex_color[1] = self.vertex_color[0]
            self.vertex_color[0] = temp

        self.ring_spacing_range = [self.vertex_radius * 3 + 0.1 * self.l, self.vertex_radius * 4 + 0.6 * self.l]
        self.unit_angle_scaling_range = [0.6, 0.8]
        self.init_angle_offset_range = [0, 1]
        self.ring_spacing_min_scaling = 0.3

    def path_from_isogeny(self, Phi):
        predetermined_path = [phi.codomain().j_invariant() for phi in Phi.factors()]
        predetermined_path = list(dict.fromkeys(predetermined_path))
        if Phi.domain().j_invariant() == predetermined_path[0]:
            predetermined_path.remove(predetermined_path[0])
        return predetermined_path

    def setup(self):
        super().setup()


    def phase_one(self, run_time=15):
        initial_dot = self.vertices[self.j_invars[0]]
        self.play(FadeIn(initial_dot))
        self.animate_path_creation(run_time=run_time)
        self.wait()

        #Traverse to common j
        end_points = (self.predetermined_paths[0][-1], self.predetermined_paths[1][-1])
        return end_points

    def phase_two(self, run_time=15):
        end_points = (self.predetermined_paths[0][-1], self.predetermined_paths[1][-1])
        PhiPrimeA = EllipticCurveHom_composite(self.EB, self.PhiB(self.A))
        PhiPrimeB = EllipticCurveHom_composite(self.EA, self.PhiA(self.B))
        self.predetermined_paths[0].append(0)  # Neccesary padding 0s
        self.predetermined_paths[1].append(0)
        if self.swap_alice_bob:
            self.predetermined_paths[1].extend(self.path_from_isogeny(PhiPrimeA))
            self.predetermined_paths[0].extend(self.path_from_isogeny(PhiPrimeB))
        else:
            self.predetermined_paths[0].extend(self.path_from_isogeny(PhiPrimeA))
            self.predetermined_paths[1].extend(self.path_from_isogeny(PhiPrimeB))
        self.paths[0].append(end_points[1])  # End points added
        self.paths[1].append(end_points[0])
        self.vertices_in_paths[0].append(self.vertices[end_points[1]])
        self.vertices_in_paths[1].append(self.vertices[end_points[0]])
        self.path_lengths = [len(self.predetermined_paths[0]), len(self.predetermined_paths[1])]
        self.animate_path_creation(run_time=run_time)
        self.wait()
        return self.predetermined_paths[0][-1]

    def walk_up(self ,initial_path_lengths):
        center, w, h = self.camera.get_center().copy(), self.camera.get_width(), self.camera.get_height()
        self.play(
            self.camera.animate.scale_to_fit_width(25).move_to(ORIGIN)
        )
        self.wait()
        run_time_per = 10 / self.path_lengths[0]
        for i in range(initial_path_lengths[0]):
            self.play(self.camera.animate.move_to(self.vertices_in_paths[0][i]), run_time=run_time_per,
                      rate_func=rate_functions.linear)
        for i in range(initial_path_lengths[1] + 2, self.path_lengths[1]):
            self.play(self.camera.animate.move_to(self.vertices_in_paths[1][i]), run_time=run_time_per,
                      rate_func=rate_functions.linear)
        self.wait(2)
        for i in range(initial_path_lengths[1]):
            self.play(self.camera.animate.move_to(self.vertices_in_paths[1][i]), run_time=run_time_per,
                      rate_func=rate_functions.linear)
        for i in range(initial_path_lengths[0] + 2, self.path_lengths[0]):
            self.play(self.camera.animate.move_to(self.vertices_in_paths[0][i]), run_time=run_time_per,
                      rate_func=rate_functions.linear)

        self.wait(2)
        self.play(self.camera.animate.scale_to_fit_width(w).move_to(center))
        self.wait()

    def construct(self):
        self.add(self.label)
        self.phase_one()
        initial_path_lengths = [len(self.paths[0]), len(self.paths[1])]

        ssk = self.phase_two()
        shared_vert = self.vertices[ssk]
        self.play(shared_vert.animate.set_color(YELLOW))
        self.walk_up(initial_path_lengths)
        self.play(shared_vert.animate.scale_to_fit_height(self.label[0].get_height() * 1.2))
        self.wait(3)

class SinglePerspectiveScene(SIDHScene):
    def __init__(self, e2=216, e3=137, L=2):
        self.L = L
        super().__init__(e2, e3, swap_alice_bob=L == 3)
        self.main_color = self.vertex_color[0]
        self.other_color = self.vertex_color[1]
        self.main_select_color = self.vertex_select_color[0]
        self.other_select_color = self.vertex_select_color[1]
        if L==2:
            self.abrev_main = r'A'
            self.abrev_other = r'B'

            Alice = OpenGLVGroup(
                Text(r' Alice ', color=self.main_color, font_size=50),
                MathTex(fr'\ell = ', f'2')
            ).arrange(DOWN)
            Alice[1][1].set_color(self.main_color)
            Bob = OpenGLVGroup(
                Text(r' Bob ', color=self.other_color, font_size=50),
                MathTex(fr'\ell = ', f'3')
            ).arrange(DOWN)
            Bob[1][1].set_color(self.other_color)
            self.sk_main = self.nA
            self.main = Alice
            self.other = Bob
        else:
            self.abrev_main = r'B'
            self.abrev_other = r'A'
            Alice = OpenGLVGroup(
                Text(r' Alice ', color=self.other_color, font_size=50),
                MathTex(fr'\ell = ', f'2')
            ).arrange(DOWN)
            Alice[1][1].set_color(self.other_color)
            Bob = OpenGLVGroup(
                Text(r' Bob ', color=self.main_color, font_size=50),
                MathTex(fr'\ell = ', f'3')
            ).arrange(DOWN)
            Bob[1][1].set_color(self.main_color)
            self.sk_main = self.nB
            self.main = Bob
            self.other = Alice

    def setup(self):
        super().setup()
        self.add(self.label[0])

    def construct(self):
        self.play(Write(self.main.shift(LEFT)))
        # self.play(Write(self.other.shift(RIGHT)))
        self.play(self.main[1].animate.move_to(self.label[1]).align_to(self.label[0], LEFT))
        initial_dot = self.vertices[self.j_invars[0]]
        self.play(self.main[0].animate.next_to(initial_dot, UP))

        if self.p > 9999999:
            num_tex, chosen = choose_random_int(self, 1, self.L**self.e2, predetermined_outcome=self.sk_main, width=6)
        else:
            num_tex, chosen = choose_random_int(self, 1, self.L**self.e2, predetermined_outcome=self.sk_main)
        sk_tex = MathTex(rf'sk_{self.abrev_main} =').next_to(num_tex, LEFT)
        self.play(Write(sk_tex))
        sk_long_str = []
        for bi in to_base(chosen, self.L):
            sk_long_str.append(str(bi))
        extra_sk = MathTex(rf'\mapsto ', *sk_long_str).scale_to_fit_height(num_tex.get_height()).next_to(num_tex, RIGHT)
        self.play(Write(extra_sk))
        OpenGLVGroup(*extra_sk[1:]).set_color(self.main_color)
        self.wait()
        sk_gp = OpenGLVGroup(sk_tex, num_tex, extra_sk)
        self.play(sk_gp.animate.arrange(RIGHT).to_corner(DL), self.main[0].animate.to_corner(DL).shift(UP*0.6))
        self.num_paths = 1
        self.label = OpenGLVGroup(
            self.label[0],
            self.main,
            sk_gp
        )

        pk1, pk2 = self.phase_one(run_time=10)
        good_height = self.vertices[pk1].get_height()
        self.play(self.vertices[pk1].animate.scale_to_fit_height(self.label[0].get_height() * 3))
        self.play(Circumscribe(self.vertices[pk1]))
        self.main = self.main[0]
        self.other.scale_to_fit_width(self.main.get_width()).align_to(self.camera.get_edge_center(RIGHT), RIGHT).shift(LEFT * self.other.get_width())
        pkM = MathTex(rf'pk_{self.abrev_main} = j(E_{self.abrev_main})', r' = ').scale_to_fit_height(self.label[0].get_height()).next_to(self.main, UP).align_to(self.main, LEFT)
        real_pkM = MathTex(latex(pk1), color=self.main_select_color).scale_to_fit_width(
            self.label[0].get_width() * 1.3).next_to(pkM, RIGHT, buff=3)
        real_pkO = MathTex(latex(pk2), color=self.other_select_color).scale_to_fit_width(
            self.label[0].get_width() * 1.3).next_to(self.other, UP).align_to(self.camera.get_edge_center(RIGHT), RIGHT).shift(LEFT * self.label[0].get_width() *0.1)
        pkO = MathTex(rf'pk_{self.abrev_other} = j(E_{self.abrev_other})', r' = ').scale_to_fit_height(
            self.label[0].get_height() * 1.3).next_to(real_pkO, LEFT, buff=3)
        self.wait()
        self.play(Write(pkM))
        self.play(GrowFromPoint(real_pkM, point=self.vertices[pk1].get_center()))
        self.play(self.vertices[pk1].animate.scale_to_fit_height(good_height))
        self.wait()
        self.play(Write(self.other))
        self.play(Write(pkO))
        self.play(Write(real_pkO))
        self.wait()
        self.play(
            FadeOut(pkM, pkO), real_pkM.animate.next_to(self.other, UP),real_pkO.animate.next_to(self.main, UP).align_to(self.main, LEFT))
        self.wait()
        self.play(FadeOut(self.other, real_pkM))
        self.play(Circumscribe(real_pkO))
        # Pick random screen point as center:
        x = (random() - 0.4) * 0.8 * self.camera.get_width() * RIGHT
        y = (random() - 0.4) * 0.8 * self.camera.get_height() * UP
        public_vert = self.j_vertex(pk2, vertex_color=self.other_select_color).shift(
            x + y).scale(1.5)
        self.vertices[pk2] = public_vert
        public_vert.scale_to_fit_height(self.label[0].get_height() * 3)
        self.play(GrowFromPoint(public_vert, point=real_pkO.get_center()), ShrinkToCenter(real_pkO))
        self.play(public_vert.animate.scale_to_fit_height(good_height))

        ssk = self.phase_two(run_time=10)

        shared_vert = self.vertices[ssk]
        self.play(shared_vert.animate.scale_to_fit_height(self.label[0].get_height() * 3).set_color(YELLOW))
        ssk_tex = MathTex(latex(ssk), color=YELLOW).scale_to_fit_width(
            self.camera.get_height() * 0.7).move_to(self.camera.get_center() + UP * self.camera.get_height() * 0.1)
        self.play(FadeOut(sk_gp))
        self.other.align_to(self.camera.get_edge_center(RIGHT), RIGHT).shift(LEFT * self.other.get_width())
        self.play(FadeIn(self.other[0].align_to(ssk_tex, UP)), self.main.animate.align_to(ssk_tex, UP))
        self.play(GrowFromPoint(ssk_tex, point=shared_vert.get_center()), shared_vert.animate.scale_to_fit_height(good_height))
        self.wait(3)



paramsets = {
            # 12: [5, 4],
            # 434: [216, 137],
            # 503: [250, 159],
            610: [305, 192],
            751: [372, 239],
        }
filenames = ['BothSIKEPerspectives', 'AliceSIKEPerspective', 'BobSIKEPerspective']

import os
import sys
import psutil
import logging





if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        filename = filenames[0]
        key = list(paramsets)[0]
        i = 0
    else:
        filename = sys.argv[2]
        key = list(paramsets)[int(sys.argv[3])]
        i = int(sys.argv[1])
    manim_configuration = {
        "output_file": filename + str(key),
        "quality": "high_quality", "preview": True,
        "disable_caching": True, "renderer": "opengl", "write_to_movie": True,
        # "from_animation_number": 0, "upto_animation_number": 70, "show_file_in_browser": False,
    }
    with tempconfig(manim_configuration):
        set_random_seed(12)
        e2, e3 = paramsets[key][0], paramsets[key][1]
        if i == 0:
            scene = SIDHScene(e2, e3)
            scene.render()
        elif i == 1:
            scene = SinglePerspectiveScene(e2, e3, L=2)
            scene.render()

        elif i == 2:
            scene = SinglePerspectiveScene(e2, e3, L=3)
            scene.render()

        python = sys.executable
        new_args = sys.argv
        if len(new_args) == 1:
            new_args.append(0)
            new_args.append(0)
            new_args.append(str(0))
        if i < 2:
            new_args[1] = str(i + 1)
            new_args[2] = filenames[i + 1]
            os.execl(python, python, *new_args)
        else:
            new_args[1] = str(0)
            new_args[2] = filenames[0]
            if int(new_args[3]) != len(list(paramsets))-1:
                new_args[3] = str(int(new_args[3]) + 1)
                os.execl(python, python, *new_args)