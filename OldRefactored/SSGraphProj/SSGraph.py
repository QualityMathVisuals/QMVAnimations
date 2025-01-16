from exceptiongroup import catch
from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup
from manim.renderer.opengl_renderer import OpenGLCamera
from sage.all import *
from sage.rings.factorint import factor_trial_division
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial

from Utilities import to_base


class BreadthFirstSSGraphScene(MovingCameraScene):
    def __init__(self, p: int, l=2):
        super().__init__()
        self.label = None
        self.p = p
        self.l = l
        self.vertex_radius = 0.5
        self.max_level_length = 10000000
        # Ring spacing is always minimum ring_spacing_range[0]. Then scaled current ring * range
        self.ring_spacing_range = [self.vertex_radius * 2 + 0.1 * self.l, self.vertex_radius * 2 + 0.6 * self.l]
        self.ring_spacing_min_scaling = 0.3
        self.unit_angle_scaling_range = [0.5, 1]
        self.init_angle_offset_range = [0, 1]
        self.use_scientific_notation = True
        self.vertex_select_color = GREEN_B
        self.vertex_color = BLUE_B

        if not is_prime(p):
            raise Exception("p must be prime")
        Fq = FiniteField(p ** 2, 'w'); w = Fq.gen()
        self.Fq = Fq
        if p % 4 == 3:
            E0 = EllipticCurve(Fq, [1, 0])
        elif p % 3 == 2:
            E0 = EllipticCurve(Fq, [0, 1])
        else:
            raise Exception("p must be congrugent to 3 mod 4 or 2 mod 3")
        j0 = E0.j_invariant()

        self.levels = {0: [j0]}
        self.j_invars = [j0]
        self.vertices = {j0: self.j_vertex(j0)}
        self.edges = {}
        self.current_level = 0


    def j_vertex(self, j_inv, vertex_color=BLUE_B):
        if j_inv == self.j_invars[0]:
            vertex_color = WHITE
        dot = Circle(radius=self.vertex_radius, color=vertex_color, fill_opacity=0.4)
        label_str = latex(j_inv)
        if self.use_scientific_notation and len(label_str) > 17:
            try:
                label_str = f'{j_inv.to_integer():.2E}'
            except:
                import decimal
                d = decimal.Decimal(int(j_inv.to_integer()))
                label_str = format(d, '.2e')
            label_str = label_str[:4] + r'\cdot 10^{' + label_str[6:] + '}'
        label = MathTex(label_str).move_to(dot.get_center())
        if label.get_width() > dot.get_width():
            label.scale_to_fit_width(dot.get_width() * 0.9)

        return OpenGLVGroup(dot, label)


    def j_edge(self, j1, j2):
        v1 = self.vertices[j1][0]; v2 = self.vertices[j2][0]
        direction_vec = v2.get_center() - v1.get_center()
        return Line(
                start=v1.get_boundary_point(direction_vec),
                end=v2.get_boundary_point(-direction_vec)
            )


    def prime_label(self):
        p = self.p
        prime_string = None
        if p < 100000:
            prime_string = str(int(p))
        # Attempt to factor p + 1 with small factors
        else:
            factorization = factor_trial_division(p + 1, 10000)
            for factor, mult in  list(factorization):
                if factor >= 100:
                    #If no good rep, just write entire prime
                    prime_string = p
            if prime_string is None:
                prime_string = latex(factorization) + r' - 1'

        return MathTex('p = ' + prime_string)


    def l_label(self):
        return MathTex(fr'\ell = {self.l}')


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
        X0j = classical_modular_polynomial(l, j_curr)
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


    def draw_next_level(self, animate_each_node=True, run_time=3):
        next_ring_js = []
        next_ring_dots = []
        next_ring_edges = {}
        N = len(self.levels[self.current_level])
        if N == 0:
            return False

        vert_runtime = run_time / N
        for j_curr in self.levels[self.current_level]:
            neighbor_js, neighboring_vertices, neighboring_edges, _ = self.draw_neighbors(j_curr, animate=animate_each_node, run_time=vert_runtime)
            next_ring_js.extend(neighbor_js)
            next_ring_dots.extend(neighboring_vertices)
            next_ring_edges = next_ring_edges | neighboring_edges

        if not animate_each_node:
            if len(next_ring_dots) > 0:
                self.play(*[
                    self.vertices[dot][0].animate.set_color(self.vertex_select_color)
                    for dot in self.levels[self.current_level]
                ], run_time=run_time / 6)
            if len(next_ring_edges) > 0:
                self.play(*[
                    Create(edge)
                    for edge in next_ring_edges.values()
                ], run_time=run_time / 3)
            if len(next_ring_dots) > 0:
                self.play(*[
                    Create(dot)
                    for dot in next_ring_dots
                ], run_time=run_time / 3)
                self.play(*[
                    self.vertices[dot][0].animate.set_color(self.vertex_color)
                    for dot in self.levels[self.current_level]
                ], run_time=run_time / 6)


        self.current_level += 1
        self.levels[self.current_level] = next_ring_js

        return len(next_ring_dots) >= len(next_ring_edges) / self.l


    def zoom_out_one_level(self, level):
        avg_scaling_factor = ((1 - self.ring_spacing_min_scaling) / 2) + self.ring_spacing_min_scaling
        avg_edge_length = self.ring_spacing_range[0] + min(level, self.max_level_length) * avg_scaling_factor * (
                self.ring_spacing_range[1] - self.ring_spacing_range[0])
        scaling_factor = (self.camera.get_height() + avg_edge_length * 2) / self.camera.get_height()
        self.play(
            self.camera.animate.set_height(self.camera.get_height() * scaling_factor),
            self.camera.animate.set_width(self.camera.get_width() * scaling_factor),
            self.label.animate.scale(scaling_factor, about_point=ORIGIN)
        )


    def setup(self):
        self.label = OpenGLVGroup(
            self.prime_label(), self.l_label()
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UL)


    def construct(self):
        self.play(Write(self.label))
        self.play(FadeIn(self.vertices[self.j_invars[0]]))
        max_ring = 9
        if self.p < 100000:
            max_ring = 100
        for i in range(max_ring):
            # should_expand = self.draw_next_level((i + 1) / 2)
            should_expand = self.draw_next_level(animate_each_node=False)
            if ((i > 1 and self.l == 2) or i > 0) and should_expand and i != max_ring - 1:
                self.zoom_out_one_level(self.current_level)
        self.wait(3)

        self.interactive_embed()

class TraverseSSGraphScene(BreadthFirstSSGraphScene):
    def __init__(self, p, ls=None, path_specifications=None, path_lengths=None, num_paths=1, animate_camera_frames=False):
        if path_lengths is None:
            path_lengths = [10]
        if ls is None:
            ls = [2]
        self.ls = ls
        super().__init__(p, ls[0])

        #Changes to inheritance
        self.ring_spacing_min_scaling = 0.0
        self.unit_angle_scaling_range = [0.5, 1]
        self.init_angle_offset_range = [-0.6, 2.1]
        self.max_level_length = 4

        #Logical things
        if len(ls) != num_paths:
            raise Exception(rf'Lengths of paths and specifications must match {len(ls), num_paths}')
        self.path_lengths=path_lengths
        self.num_paths=num_paths
        self.path_hashes = None
        self.predetermined_paths = None
        if path_specifications is None or len(path_specifications[0]) == 0:
            self.path_hashes = [
                [randrange(0, ls[i]) for _ in range(path_lengths[i])]
                for i in range(num_paths)
            ]
        elif path_specifications[0][0] > ls[0]:
            self.predetermined_paths = [
                path_spec for path_spec in path_specifications
            ]
        else:
            self.path_hashes = []
            for i in range(num_paths):
                if isinstance(path_specifications[i], int):
                    self.path_hashes.append(to_base(path_specifications[i], ls[i]))
                    while len(self.path_hashes[i]) < path_lengths[i]:
                        self.path_hashes[i].insert(0, 0)

                elif isinstance(path_specifications[i], list):
                    self.path_hashes.append(path_specifications[i])
                else:
                    raise Exception(f'path specification must be an integer or a list of integers:\n Gotten type {type(path_specifications[i])}')

        path_A = [self.j_invars[0]]
        self.paths = [
            path_A.copy()
            for _ in range(num_paths)
        ]

        #Render Things
        if num_paths == 1:
            self.vertex_color = [BLUE_B]
            self.vertex_select_color = [GREEN_B]
        else:
            self.vertex_color = color_gradient([BLUE_B, ORANGE], num_paths)
            self.vertex_select_color = color_gradient([GREEN_B, RED_D], num_paths)

        self.camera_frames = []
        self.animate_camera_frames = animate_camera_frames
        self.fade_factor = 0.75
        self.edges_in_paths = [
            []
            for _ in range(num_paths)
        ]
        self.vertices_in_paths = [
            []
            for _ in range(num_paths)
        ]

    def l_label(self):
        l_list_tex = [Tex(f'{self.ls[0]}', color=self.vertex_color[0])]
        for i in range(1, self.num_paths):
            l_color = self.vertex_color[i]
            l_list_tex.append(Tex(f'{self.ls[i]}', color=l_color))

        return OpenGLVGroup(MathTex(fr'\ell = '), *l_list_tex).arrange(RIGHT)

    def setup(self):
        super().setup()
        self.camera_frames.append((self.camera.get_width(), self.camera.get_height(), ORIGIN))

    def extend_path(self, path_index, animate=True, run_time=2):
        path = self.paths[path_index]
        l = self.ls[path_index]
        j_curr = path[len(path) - 1]
        dot_curr = self.vertices[j_curr]
        new_neighbor_js, neighboring_vertices, neighboring_edges, repeat_js = self.draw_neighbors(
            j_curr, l,
            vertex_select_color=self.vertex_select_color[path_index],
            vertex_color=self.vertex_color[path_index],
            animate=animate,
            run_time=run_time / 2
        )
        accessible_neighbor_js = new_neighbor_js + repeat_js
        for dot_in_list in path:
            if dot_curr in accessible_neighbor_js:
                accessible_neighbor_js.remove(dot_in_list)
        path_max = 1 + (len(self.path_hashes[path_index])
                        if self.path_hashes is not None else len(self.predetermined_paths[path_index]))
        if len(accessible_neighbor_js) == 0 or len(path) == path_max:
            return False, None, None, None, None, None, None

        if self.path_hashes is not None:
            j_next = accessible_neighbor_js[self.path_hashes[path_index][len(path) - 1] % len(accessible_neighbor_js)]

        else:
            j_next = self.predetermined_paths[path_index][len(path) - 1]
            if j_next not in accessible_neighbor_js:
                raise Exception(f'Invalid path specification. The vertex {j_curr} does not have neighbor'
                                f' {self.predetermined_paths[path_index][len(path) - 1]} or at least cannot '
                                f'travel to it without backtracking.\n Neighbors: {new_neighbor_js + repeat_js}.\n Accessible: {accessible_neighbor_js}.')

        dot_next = self.vertices[j_next]
        edge_next = neighboring_edges[j_next]

        self.vertices_in_paths[path_index].append(dot_next)
        self.edges_in_paths[path_index].append(edge_next)
        to_fade_neighbor_dots = [
            self.vertices[new_j]
            for new_j in new_neighbor_js if new_j != j_next
        ]
        to_fade_neighbor_edges = [
            neighboring_edges[j_other]
            for j_other in accessible_neighbor_js if j_other != j_next
        ]
        if animate:
            self.play(
                dot_next[0].animate.set_color(self.vertex_select_color[path_index]), run_time=run_time / 4
            )
            self.play(
                dot_curr[0].animate.set_color(self.vertex_color[path_index]),
                *[
                    dot.animate.fade(self.fade_factor)
                    for dot in to_fade_neighbor_dots
                ],*[
                    edge.animate.fade(self.fade_factor)
                    for edge in to_fade_neighbor_dots
                ],
                run_time=run_time / 4
            )

        self.paths[path_index].append(j_next)
        return True, neighboring_vertices, neighboring_edges, dot_next, edge_next, to_fade_neighbor_dots, to_fade_neighbor_edges

    def extend_paths(self, path_indices, animate_each_path=False, run_time=3.0):
        if animate_each_path:
            for i in path_indices:
                should_extend, _, _, _, _ = self.extend_path(i, animate=True, run_time=run_time / self.num_paths)
                if not should_extend:
                    path_indices.remove(i)
            return path_indices

        all_neighboring_dots = []
        all_neighboring_edges = []
        all_leading_dots = []
        all_leading_edges = []
        to_fade_dots = []
        to_fade_edges = []
        new_path_indices = path_indices.copy()
        for i in path_indices:
            should_extend, neighboring_vertices, neighboring_edges, dot_next, edge_next, to_fade_neighbor_dots, to_fade_neighbor_edges = self.extend_path(
                i, animate=False, run_time=run_time / self.num_paths)
            if not should_extend:
                new_path_indices.remove(i)
            else:
                all_neighboring_dots.extend(neighboring_vertices)
                all_neighboring_edges.extend(neighboring_edges.values())
                all_leading_dots.append(dot_next)
                all_leading_edges.append(edge_next)
                to_fade_dots.extend(to_fade_neighbor_dots)
                to_fade_edges.extend(to_fade_neighbor_edges)

        path_indices = new_path_indices
        if len(path_indices) == 0:
            return []

        if len(all_neighboring_edges) > 0:
            self.play(*[
                Create(edge)
                for edge in all_neighboring_edges
            ], run_time=run_time / 8)
        if len(all_neighboring_dots) > 0:
            self.play(*[
                Create(dot)
                for dot in all_neighboring_dots
            ], run_time=run_time / 8)
        if len(all_neighboring_edges) > 0:
            for path_index in path_indices:
                self.current_vert_updater(self.vertices[self.paths[path_index][len(self.paths[path_index]) - 1]][0], run_time=3 * run_time / 8)
            self.play(*[
                self.vertices[self.paths[path_index][len(self.paths[path_index]) - 1]][0].animate.set_color(
                    self.vertex_select_color[path_index])
                for path_index in path_indices
            ], run_time=3 * run_time / 8)
            self.play(
                *[
                     self.vertices[self.paths[path_index][len(self.paths[path_index]) - 1]][0].animate.set_color(
                         self.vertex_color[path_index])
                     for path_index in path_indices
                 ],
                *[
                    dot.animate.fade(self.fade_factor)
                    for dot in to_fade_dots
                ], *[
                    edge.animate.fade(self.fade_factor)
                    for edge in to_fade_edges
                ],
                run_time=3 * run_time / 8
            )
        return path_indices

    def zoom_out_fit_paths(self, level, run_time=1.0):
        camera_packet = self.camera_frames[len(self.camera_frames) - 1]
        final_new_width, final_new_height, final_center_shift = camera_packet[0], camera_packet[1], np.array([0.0, 0.0, 0.0])
        old_width = camera_packet[0] / 2
        old_height = camera_packet[1] / 2
        old_center = camera_packet[2]
        for path in self.paths:
            last_pos = self.vertices[path[len(path) - 1]].get_center()
            max_edge_length = self.ring_spacing_range[0] + min(level, self.max_level_length) * (
                    self.ring_spacing_range[1] - self.ring_spacing_range[0])
            buff = self.vertex_radius * (1 + 0.05 * level) + 0.5
            width_difference, height_difference = 0, 0
            width_shift, height_shift = 0, 0
            if (last_pos + RIGHT * (max_edge_length + buff))[0] > old_center[0] + final_center_shift[0] + final_new_width / 2:
                width_difference =(last_pos + RIGHT * (max_edge_length + buff))[0] - (old_center[0] + final_center_shift[0] + final_new_width / 2)
                width_shift = width_difference / 2
            elif (last_pos + LEFT * (max_edge_length + buff))[0] < old_center[0] + final_center_shift[0] - final_new_width / 2:
                width_difference = (old_center[0] + final_center_shift[0] - final_new_width / 2) - (last_pos + LEFT * (max_edge_length + buff))[0]
                width_shift = -width_difference / 2
            if (last_pos + UP * (max_edge_length + buff))[1] > old_center[1] + final_center_shift[1] + final_new_height / 2:
                height_difference =(last_pos + UP * (max_edge_length + buff))[1] - (old_center[1] + final_center_shift[1] + final_new_height / 2)
                height_shift = height_difference / 2
            elif (last_pos + DOWN * (max_edge_length + buff))[1] < old_center[1] + final_center_shift[1] - final_new_height / 2:
                height_difference = (old_center[1] + final_center_shift[1] - final_new_height / 2) - (last_pos + DOWN * (max_edge_length + buff))[1]
                height_shift = -height_difference / 2

            center_shift = np.array([width_shift, height_shift, 0])

            final_new_width += width_difference
            final_new_height += height_difference
            final_center_shift += center_shift

        if self.animate_camera_frames:
            self.add(Dot(old_center + final_center_shift, color=RED))
            self.add(Dot(self.label.get_center(), color=PINK))
            self.add(Line(start=old_center, end=old_center + final_center_shift, color=YELLOW))
            self.add(Rectangle(width=final_new_width, height=final_new_height, color=GREEN_D).shift(old_center + final_center_shift))
        scaling_factor = max(
            final_new_height / self.camera.get_height(),
            final_new_width / self.camera.get_width()
        )
        #need to fit to rectangle
        if scaling_factor > 1.0000000000000001:
            self.play(
                self.camera.animate.scale(scaling_factor, about_point=old_center).shift(final_center_shift),
                self.label.animate.scale(scaling_factor, about_point=old_center).shift(final_center_shift),
                run_time = run_time
            )
        self.camera_frames.append((final_new_width, final_new_height, old_center + final_center_shift))

    def animate_path_creation(self, run_time=10.0):
        initial_dots = list({
            # *[self.vertices[self.paths[path_index][len(self.paths[path_index]) - 1]] for path_index in range(self.num_paths)]
            *[self.vertices[self.paths[path_index][0]] for path_index in range(self.num_paths)]
        })
        initial_lengths = [
            len(self.vertices_in_paths[path_index]) for path_index in range(self.num_paths)
        ]

        # Main Path Building Animations
        path_building_run_time = run_time * 0.8
        max_path_length = max([self.path_lengths[path_index] - initial_lengths[path_index] for path_index in range(self.num_paths)])
        continuing_paths = list(range(self.num_paths))
        for _ in range(max_path_length):
            continuing_paths = self.extend_paths(continuing_paths, run_time=path_building_run_time / max_path_length)
            self.current_level += 1
            if len(continuing_paths) > 0:
                self.zoom_out_fit_paths(self.current_level + 1, path_building_run_time * 0.1 / max_path_length)
            else:
                break

        # Highlight Made Path
        path_highlighting_run_time = run_time * 0.2
        actual_max_path_length = max(
            [len(self.paths[path_index]) - initial_lengths[path_index] for path_index in range(self.num_paths)])
        larger_radius = self.vertex_radius * 1.5
        vert_run_time = path_highlighting_run_time / actual_max_path_length

        self.play(*[initial_dots[i][0].animate.set_color(
            self.vertex_select_color[i]).scale_to_fit_width(larger_radius * 2) for i in range(len(initial_dots))
        ], run_time=vert_run_time)
        for i in range(actual_max_path_length + 1):
            vertices_to_draw_with_colors = []
            visited_js = []
            for j in range(self.num_paths):
                step_of_path = i + initial_lengths[j] - 1
                if step_of_path < len(self.vertices_in_paths[j]) + 1:
                    invariant_j = self.paths[j][step_of_path]
                    if invariant_j not in visited_js:
                        visited_js.append(invariant_j)
                        vertices_to_draw_with_colors.append((self.vertices[invariant_j], self.vertex_select_color[j]))
            if len(vertices_to_draw_with_colors) > 0:
                self.play(
                    *[vertex[0].animate.scale_to_fit_width(larger_radius * 2).set_color(select_color) for vertex, select_color in
                      vertices_to_draw_with_colors],
                    run_time=vert_run_time
                )
        mobs_to_fade = []
        all_vertex_path_list = [vert for path in self.vertices_in_paths for vert in path]
        all_edge_path_list = [edge for path in self.edges_in_paths for edge in path]
        # self.play(self.edges_in_paths[0][-1].animate.scale(100))
        # self.play(self.edges_in_paths[1][-1].animate.scale(100))
        verts_to_not_fade = [self.vertices[self.j_invars[0]]]
        edges_to_not_fade = []
        for path_index in range(self.num_paths):
            last_j = self.paths[path_index][len(self.paths[path_index]) - 1]
            X0j = classical_modular_polynomial(self.ls[path_index], last_j)
            for root, mult in X0j.roots():
                if root in self.vertices.keys():
                    verts_to_not_fade.append(self.vertices[root])
                if (last_j, root) in self.edges.keys():
                    edges_to_not_fade.append(self.edges[(last_j, root)])

        for vert in self.vertices.values():
            if vert not in all_vertex_path_list and vert not in verts_to_not_fade:
                mobs_to_fade.append(vert)

        for edge in self.edges.values():
            if edge not in all_edge_path_list and edge not in edges_to_not_fade:
                mobs_to_fade.append(edge)

        self.play(FadeOut(*mobs_to_fade))
        base_vert = self.vertices[self.j_invars[0]]
        # self.edges = all_edge_path_list
        self.vertices.clear()
        init_j = self.j_invars[0]
        self.j_invars.clear()
        self.j_invars.append(init_j)
        self.vertices[self.j_invars[0]] = base_vert
        for j in range(len(self.paths)):
            i = 0
            for k in range(len(self.paths[j]) - 1):
                j_path = self.paths[j][k + 1]
                self.j_invars.append(j_path)
                self.vertices[j_path] = self.vertices_in_paths[j][i]
                i += 1

        new_edges = {}
        for key in self.edges.keys():
            if self.edges[key] in all_edge_path_list:
                new_edges[key] = self.edges[key]
        self.edges = new_edges

    def current_vert_updater(self, current_vertex, run_time):
        pass

    def construct(self):
        #IntroAnimations
        self.play(Write(self.label))
        initial_dot = self.vertices[self.j_invars[0]]
        self.play(FadeIn(initial_dot))
        self.animate_path_creation(run_time=20)

        self.wait()
        self.interactive_embed()

manim_configuration = {"quality": "high_quality", "preview": True, "disable_caching": True, "output_file": 'ManyWalksBigPrimel3',
                       # "from_animation_number": 0, "upto_animation_number": 70,
                       "renderer": "opengl", "write_to_movie": True, "show_file_in_browser": False, }
if __name__ == '__main__':
    with tempconfig(manim_configuration):
        p = (2**216)*(3**137)-1
        # p = 2063
        # scene = BreadthFirstSSGraphScene(p, l=11)
        scene = TraverseSSGraphScene(p, ls=[5 for i in range(10)], path_specifications=None, path_lengths=[50 for i in range(10)], num_paths=10, animate_camera_frames=False)
        scene.render()
