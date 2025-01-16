from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union
from manimlib import VGroup, ParametricCurve
import numpy as np
from scipy.interpolate import Rbf
import re

class WolframAlphaCurve(VGroup):
    def __init__(self, xt: str, yt: str, x_range: Tuple[float, float], y_range: Tuple[float, float], t_ranges: Tuple[float, float], max_dist_from_center: float = 4, **kwargs):
        self.xt = xt
        self.yt = yt
        self.x_range = x_range
        self.y_range = y_range
        self.t_ranges = t_ranges
        self.max_dist_from_center = max_dist_from_center
        self.param_func = self._create_param_function()
        super().__init__(
            ParametricCurve(self.param_func, t_range=t_range, **kwargs)
            for t_range in self.t_ranges
        )

    def _string_to_function(self, expression: str) -> Callable:
        # Standardize spacing and remove all spaces
        expr = re.sub(r'\s+', '', expression.strip())
        
        # Add multiplication operators between various elements
        expr = re.sub(r'(\d+)([st])', r'\1*\2', expr)  # Number and variables/functions
        expr = re.sub(r'(\d+)\(', r'\1*(', expr)  # Number and parentheses
        expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)  # Parentheses and functions
        expr = re.sub(r'\)(\d)', r')*\1', expr)  # Parentheses and numbers
        expr = re.sub(r'(\d+)π', r'\1*π', expr)  # Numbers and pi
        
        # Handle special cases
        cutoff_at_theta = lambda s: s[:re.search(r"θ\(sqrt\(sgn\(", s).start()] if re.search(r"θ\(sqrt\(sgn\(", s) else s
        expr = cutoff_at_theta(expr)
        
        # Replace mathematical functions
        replacements = {
            'sin': 'np.sin',
            'cos': 'np.cos',
            'sqrt': 'np.sqrt',
            'θ': 'heaviside',
            'π': 'np.pi'
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # Add multiplication operators for heaviside
        expr = re.sub(r'\)heaviside', r')*heaviside', expr)

        def generated_function(t):
            def heaviside(x):
                return np.heaviside(x, 0.5)
            return eval(expr, {"np": np, "heaviside": heaviside}, {"t": t})

        return generated_function

    def _get_max_range(self, x_range, y_range) -> float:
        return max(abs(val) for lst in [x_range, y_range] for val in lst)

    def _create_param_function(self) -> Callable:
        x_func = self._string_to_function(self.xt)
        y_func = self._string_to_function(self.yt)
        scaling_factor = self.max_dist_from_center / self._get_max_range(self.x_range, self.y_range)
        return lambda t: (
            scaling_factor * x_func(t),
            scaling_factor * y_func(t),
            0
        )


class FacePointData:
    def __init__(self, outline, hair, ears, eyes, mouth, nose, extras, add_extra_outline=True):
        self.outline = outline
        if add_extra_outline:
            distance_threshold = 0.1
            self.extra_outline = [
                (x * 1.1, y * 1.1, 0)
                for x, y, _ in outline
                if not any(np.linalg.norm(np.array((x * 1.1, y * 1.1)) - np.array((ear_x, ear_y))) < distance_threshold for ear_x, ear_y, _ in ears)
            ] 
            self.extra_outline += [
                (x * 1.3, y * 1.3, 0)
                for x, y, _ in ears
                if not any(np.linalg.norm(np.array((x * 1.1, y * 1.1)) - np.array((out_x, out_y))) < distance_threshold for out_x, out_y, _ in outline)
            ]
        else:
            self.extra_outline = []
        self.hair = hair
        self.ears = ears
        self.eyes = eyes
        self.mouth = mouth
        self.nose = nose
        self.extras = extras

    def __repr__(self):
        """
        Provide a string representation for debugging.
        """
        return (
            f"FacePointData(\n"
            f"  outline={len(self.outline)} points,\n"
            f"  extra_outline={len(self.extra_outline)} points,\n"
            f"  hair={len(self.hair)} points,\n"
            f"  ears={len(self.ears)} points,\n"
            f"  eyes={len(self.eyes)} points,\n"
            f"  mouth={len(self.mouth)} points,\n"
            f"  nose={len(self.nose)} points,\n"
            f"  extras={len(self.extras)} points\n"
            f")"
        )

    def get_all_points(self):
        return (
            self.outline +
            self.extra_outline +
            self.hair +
            self.ears +
            self.eyes +
            self.mouth +
            self.nose +
            self.extras
        )

    def get_points_as_list(self):
        return [
            self.outline,
            self.extra_outline,
            self.hair,
            self.ears,
            self.eyes,
            self.mouth,
            self.nose,
            self.extras,
        ]

    def get_all_points_but_extra_outline(self):
        return (
            self.outline +
            self.hair +
            self.ears +
            self.eyes +
            self.mouth +
            self.nose +
            self.extras
        )


@staticmethod
#TODO finish
def _generate_face_fixed_outline(rim_outline_points, other_points):
    """Generates a surface function based on input points and a fixed rim at a given radius."""
    x, y, z = zip(*other_points)
    # Add points on the fixed rim to enforce z=0 at the given radius
    x_rim, y_rim, _ = zip(*rim_outline_points)
    z_rim = np.zeros_like(x_rim)
    # Combine original points with rim points
    x_combined = np.concatenate((x, x_rim))
    y_combined = np.concatenate((y, y_rim))
    z_combined = np.concatenate((z, z_rim))

    # Create RBF interpolator
    rbf = Rbf(x_combined, y_combined, z_combined, smooth=0.1)

    def points_to_surface(x, y):
        return rbf(x, y)

    return points_to_surface

@staticmethod
def _generate_face_fixed_radius(radius, points):
    """Generates a surface function based on input points and a fixed rim at a given radius."""
    x, y, z = zip(*points)
    # Add points on the fixed rim to enforce z=0 at the given radius
    num_rim_points = len(points)  # Number of points to approximate the rim
    theta_rim = np.linspace(0, 2 * np.pi, num_rim_points, endpoint=False)
    x_rim = radius * np.cos(theta_rim)
    y_rim = radius * np.sin(theta_rim)
    z_rim = np.zeros_like(theta_rim)
    # Combine original points with rim points
    x_combined = np.concatenate((x, x_rim))
    y_combined = np.concatenate((y, y_rim))
    z_combined = np.concatenate((z, z_rim))

    # Create RBF interpolator
    rbf = Rbf(x_combined, y_combined, z_combined, smooth=0.15)

    # Define the function that maps (r, theta) -> z
    def points_to_surface(r, theta):
        """Maps polar coordinates (r, theta) to z using the interpolated surface."""
        x_query = r * np.cos(theta)
        y_query = r * np.sin(theta)
        return rbf(x_query, y_query)

    return points_to_surface

@staticmethod
def _generate_face_fixed_rectangle(height, width, points):
    """Generates a surface function based on input points and a fixed rim at a given radius."""
    x, y, z = zip(*points)
    # Add points on the fixed rim to enforce z=0 at the given radius
    num_rim_points = len(points)  # Number of points to approximate the rim
    num_side_points = num_rim_points // 4
    x_rim = np.concatenate([
        np.linspace(-width / 2, width / 2, num_side_points, endpoint=False),
        np.full(num_side_points, width / 2),
        np.linspace(width / 2, -width / 2, num_side_points, endpoint=False),
        np.full(num_side_points, -width / 2)
    ])
    y_rim = np.concatenate([
        np.full(num_side_points, -height / 2),
        np.linspace(-height / 2, height / 2, num_side_points, endpoint=False),
        np.full(num_side_points, height / 2),
        np.linspace(height / 2, -height / 2, num_side_points, endpoint=False)
    ])
    z_rim = np.zeros_like(x_rim)
    # Combine original points with rim points
    x_combined = np.concatenate((x, x_rim))
    y_combined = np.concatenate((y, y_rim))
    z_combined = np.concatenate((z, z_rim))

    # Create RBF interpolator
    rbf = Rbf(x_combined, y_combined, z_combined, smooth=0.15)

    def points_to_surface(x, y):
        return rbf(x, y)

    return points_to_surface

@staticmethod
def _get_component_points(param_func: Callable, 
                       t_ranges: List[Tuple[float, float, float]], 
                       t_range_indices: List[int], 
                       num_points: int = 10, 
                       z_val: float = 0) -> List[Tuple[float, float, float]]:
    per_component_pts = num_points // len(t_range_indices)
    points = []
    for i in t_range_indices:
        t_start, t_end, _ = t_ranges[i]
        ts = np.linspace(t_start, t_end, int(per_component_pts))
        xs, ys, _ = param_func(ts)
        points.extend(zip(xs, ys, z_val + np.zeros_like(xs)))
    return points


class HumanFace(ABC):
    def __init__(self, wolfram_curve, resolution: float = 1):
        self.resolution = resolution
        self.curve = wolfram_curve

    @abstractmethod
    def get_3D_points_on_face(self) -> FacePointData:
        pass

    def get_3D_param_radial(self, radius):
        return _generate_face_fixed_radius(radius, self.get_3D_points_on_face().get_all_points())

    def get_3D_param_rectangular(self, height, width):
        return _generate_face_fixed_rectangle(height, width, self.get_3D_points_on_face().get_all_points())

    def get_3D_param_faceoutline(self):
        face_data = self.get_3D_points_on_face()
        return _generate_face_fixed_outline(face_data.extra_outline, face_data.get_all_points_but_extra_outline())

class ObamaFace(HumanFace):
    def __init__(self, resolution: float = 1, **kwargs):
        tau = 2 * np.pi
        t_ranges = [
            (i * tau, (i + 1) * tau, 0.1 * (1/resolution)) 
            for i in range(0, 30, 2)
        ]

        curve = WolframAlphaCurve(
            xt = '((-6/7 sin(22/15 - 13 t) - 46/45 sin(11/10 - 10 t) - 13/7 sin(16/11 - 3 t) - 69/10 sin(17/12 - 2 t) + 529/5 sin(t + 11/7) + 4/9 sin(4 t + 38/9) + 32/7 sin(5 t + 10/7) + 4/3 sin(6 t + 13/14) + 25/4 sin(7 t + 16/11) + 9/7 sin(8 t + 10/11) + 43/10 sin(9 t + 17/12) + 3/7 sin(11 t + 22/5) + 4/9 sin(12 t + 20/9) + 81/5) θ(59 π - t) θ(t - 55 π) + (-251/13 sin(11/8 - 3 t) + 4100/11 sin(t + 8/5) + 31/9 sin(2 t + 9/4) + 17/12 sin(4 t + 22/5) + 115/9 sin(5 t + 22/13) + 24/11 sin(6 t + 9/2) + 201/10 sin(7 t + 17/10) + 9/4 sin(8 t + 13/5) + 112/11) θ(55 π - t) θ(t - 51 π) + (507/5 sin(t + 11/7) + 1/5 sin(2 t + 48/11) + 73/10 sin(3 t + 11/7) + 23/10 sin(4 t + 11/7) + 1359/7) θ(51 π - t) θ(t - 47 π) + (680/7 sin(t + 11/7) + 8/13 sin(2 t + 23/15) + 51/8 sin(3 t + 11/7) + 1/52 sin(4 t + 19/7) + 11/5 sin(5 t + 8/5) - 497/3) θ(47 π - t) θ(t - 43 π) + (-4/11 sin(1 - 7 t) - 303/8 sin(11/7 - t) + 85/6 sin(2 t + 14/9) + 9/8 sin(3 t + 19/14) + 126/25 sin(4 t + 23/15) + 13/12 sin(5 t + 21/16) + 12/11 sin(6 t + 19/14) + 2/3 sin(8 t + 10/7) + 2/5 sin(9 t + 29/19) - 2219/13) θ(43 π - t) θ(t - 39 π) + (-3/8 sin(14/9 - 12 t) - 3/7 sin(11/7 - 10 t) - 1/2 sin(14/9 - 8 t) - 52/21 sin(14/9 - 6 t) - 19/6 sin(17/11 - 4 t) - 47/11 sin(11/7 - 3 t) - 132/7 sin(11/7 - 2 t) + 229/6 sin(t + 11/7) + 3/11 sin(5 t + 13/9) + 1/4 sin(7 t + 37/25) + 1/11 sin(9 t + 13/8) + 5/11 sin(11 t + 11/7) + 1/6 sin(13 t + 23/15) + 2209/11) θ(39 π - t) θ(t - 35 π) + (-1/2 sin(14/9 - 4 t) + 1037/8 sin(t + 14/9) + 2/11 sin(2 t + 41/11) + 89/5 sin(3 t + 17/11) + 71/14 sin(5 t + 11/7) + 115/7) θ(35 π - t) θ(t - 31 π) + (-137/69 sin(2/5 - 6 t) + 679/13 sin(t + 8/7) + 355/59 sin(2 t + 60/13) + 70/11 sin(3 t + 16/7) + 19/6 sin(4 t + 5/2) + 49/13 sin(5 t + 20/7) + 9/11 sin(7 t + 173/43) + 7/8 sin(8 t + 19/14) - 3897/10) θ(31 π - t) θ(t - 27 π) + (-7/10 sin(3/8 - 8 t) - 12/11 sin(2/9 - 6 t) - 23/7 sin(3/8 - 2 t) + 1171/8 sin(t + 14/9) + 71/6 sin(3 t + 17/11) + 13/7 sin(4 t + 6/13) + 25/4 sin(5 t + 3/2) + 34/9 sin(7 t + 18/11) + 27/10 sin(9 t + 4/3) + 71/5) θ(27 π - t) θ(t - 23 π) + (-3/2 sin(3/4 - 2 t) + 127/4 sin(t + 10/7) - 3329/21) θ(23 π - t) θ(t - 19 π) + (344/11 sin(t + 37/8) + 24/11 sin(2 t + 23/10) + 1694/9) θ(19 π - t) θ(t - 15 π) + (-1/9 sin(9/19 - 10 t) - 1/7 sin(1/16 - 8 t) - 1/13 sin(4/7 - 6 t) - 11/13 sin(36/37 - 4 t) - 20/9 sin(10/11 - 2 t) + 998/9 sin(t + 9/5) + 77/8 sin(3 t + 7/3) + 20/7 sin(5 t + 36/13) + 4/3 sin(7 t + 13/4) + 11/15 sin(9 t + 60/17) + 1036/5) θ(15 π - t) θ(t - 11 π) + (-6/19 sin(11/8 - 12 t) + 760/7 sin(t + 12/7) + 38/11 sin(2 t + 26/17) + 73/8 sin(3 t + 35/17) + 17/12 sin(4 t + 65/32) + 21/8 sin(5 t + 16/7) + 5/9 sin(6 t + 10/3) + 24/25 sin(7 t + 15/7) + 8/17 sin(8 t + 32/9) + 2/7 sin(9 t + 19/10) + 3/10 sin(10 t + 47/11) + 5/16 sin(11 t + 13/10) - 3010/17) θ(11 π - t) θ(t - 7 π) + (-11/14 sin(1/6 - 8 t) - 12/5 sin(1/5 - 4 t) - 48/7 sin(39/40 - 2 t) + 15/14 sin(7 t) + 959/20 sin(t + 29/9) + 112/11 sin(3 t + 13/8) + 25/9 sin(5 t + 14/9) + 5/11 sin(6 t + 11/17) + 6/7 sin(9 t + 25/6) + 5425/13) θ(7 π - t) θ(t - 3 π) + (-8/17 sin(9/11 - 21 t) - 24/25 sin(3/5 - 20 t) - 3/4 sin(17/16 - 14 t) - 15/7 sin(11/21 - 7 t) - 5/8 sin(31/32 - 6 t) - 63/16 sin(4/5 - 5 t) + 2153/5 sin(t + 19/12) + 230/7 sin(2 t + 33/10) + 69/2 sin(3 t + 61/13) + 32/3 sin(4 t + 37/12) + 20/9 sin(8 t + 31/8) + 36/11 sin(9 t + 12/11) + 9/8 sin(10 t + 23/5) + 24/11 sin(11 t + 32/7) + 25/13 sin(12 t + 29/9) + 8/9 sin(13 t + 35/17) + 9/10 sin(15 t + 2) + 2/11 sin(16 t + 7/8) + 1/2 sin(17 t + 26/7) + 7/12 sin(18 t + 29/10) + 7/5 sin(19 t + 1) + 10/9 sin(22 t + 2) + 4/11 sin(23 t + 2/7) + 3/5 sin(24 t + 5/2) + 322/17) θ(3 π - t) θ(t + π)) θ(sqrt(sgn(sin(t/2))))',
            yt = '((-22/9 sin(34/23 - 9 t) - 61/9 sin(17/11 - 4 t) + 37/11 sin(t + 65/14) + 41/6 sin(2 t + 14/3) + 8/7 sin(3 t + 15/7) + 18/5 sin(5 t + 7/5) + 91/8 sin(6 t + 25/17) + 16/11 sin(7 t + 5/16) + 79/14 sin(8 t + 25/17) + 3/8 sin(10 t + 18/11) + 2/5 sin(11 t + 57/14) + 1/4 sin(12 t + 27/13) + 5/6 sin(13 t + 22/5) - 2434/7) θ(59 π - t) θ(t - 55 π) + (-30/11 sin(11/10 - 5 t) - 143/3 sin(3/2 - 4 t) - 108/31 sin(1/6 - 3 t) - 1310/9 sin(17/11 - 2 t) + 57/8 sin(t + 9/7) + 25/17 sin(6 t + 25/13) + 3/10 sin(7 t + 31/21) + 23/14 sin(8 t + 51/11) + 635/3) θ(55 π - t) θ(t - 51 π) + (-40/9 sin(11/7 - 4 t) - 49/2 sin(11/7 - 2 t) - 188/17 sin(11/7 - t) + 26/9 sin(3 t + 11/7) - 1143/13) θ(51 π - t) θ(t - 47 π) + (-25/4 sin(11/7 - 4 t) + 93/11 sin(t + 11/7) + 290/11 sin(2 t + 33/7) + 5/6 sin(3 t + 33/7) + 10/11 sin(5 t + 14/9) - 262/3) θ(47 π - t) θ(t - 43 π) + (-3/2 sin(14/9 - 7 t) + 286/3 sin(t + 33/7) + 71/15 sin(2 t + 50/11) + 253/21 sin(3 t + 37/8) + 11/12 sin(4 t + 36/13) + 47/14 sin(5 t + 32/7) + 4/5 sin(6 t + 9/5) + 2/9 sin(8 t + 14/11) + 11/7 sin(9 t + 47/10) - 2032/5) θ(43 π - t) θ(t - 39 π) + (-3/7 sin(35/23 - 13 t) - 3/8 sin(14/9 - 10 t) - 9/7 sin(17/11 - 9 t) - 1/16 sin(5/11 - 8 t) - 35/17 sin(14/9 - 7 t) - 3/7 sin(26/17 - 6 t) - 37/13 sin(11/7 - 5 t) - 109/9 sin(14/9 - 3 t) - 24/5 sin(17/11 - 2 t) - 833/9 sin(11/7 - t) + 4/3 sin(4 t + 14/9) + 11/15 sin(11 t + 33/7) + 1/27 sin(12 t + 7/8) - 3683/9) θ(39 π - t) θ(t - 35 π) + (3/5 sin(t + 35/23) + 22/9 sin(2 t + 14/9) + 23/12 sin(3 t + 61/13) + 37/6 sin(4 t + 14/3) + 12/13 sin(5 t + 14/9) - 1463/3) θ(35 π - t) θ(t - 31 π) + (-148/11 sin(1/15 - 3 t) - 3/7 sin(3/13 - 2 t) - 1356/11 sin(17/18 - t) + 19/6 sin(4 t + 33/16) + 245/61 sin(5 t + 9/11) + 5/6 sin(6 t + 63/16) + 10/7 sin(7 t + 13/8) + 9/8 sin(8 t + 25/9) - 2513/12) θ(31 π - t) θ(t - 27 π) + (-348/7 sin(1/37 - t) + 65/11 sin(2 t + 20/13) + 34/11 sin(3 t + 32/11) + 59/17 sin(4 t + 33/8) + 105/19 sin(5 t + 31/10) + 24/23 sin(6 t + 6/11) + 15/16 sin(7 t + 31/30) + 4/9 sin(8 t + 85/21) + 1/3 sin(9 t + 41/11) - 4459/9) θ(27 π - t) θ(t - 23 π) + (240/11 sin(t + 1/8) + 23/10 sin(2 t + 23/12) - 997/11) θ(23 π - t) θ(t - 19 π) + (-380/17 sin(1/11 - t) + 20/11 sin(2 t + 20/19) - 625/7) θ(19 π - t) θ(t - 15 π) + (-67/15 sin(9/11 - 4 t) - 66/5 sin(8/9 - 2 t) - 360/13 sin(4/13 - t) + 28/5 sin(3 t + 9/13) + 16/5 sin(5 t + 7/6) + 1/2 sin(6 t + 12/13) + 13/9 sin(7 t + 5/3) + 5/8 sin(8 t + 8/7) + 10/9 sin(9 t + 40/17) + 10/19 sin(10 t + 61/20) - 28) θ(15 π - t) θ(t - 11 π) + (-1/15 sin(8/11 - 10 t) - 3/14 sin(3/8 - 8 t) - 49/50 sin(5/11 - 6 t) - 28/9 sin(11/9 - 4 t) - 62/5 sin(6/5 - 2 t) + 169/8 sin(t + 12/25) + 149/15 sin(3 t + 60/59) + 40/17 sin(5 t + 10/9) + 8/5 sin(7 t + 13/10) + 14/11 sin(9 t + 18/11) + 7/11 sin(11 t + 7/5) + 2/5 sin(12 t + 16/9) - 86/3) θ(11 π - t) θ(t - 7 π) + (-9/11 sin(9/14 - 9 t) + 995/8 sin(t + 38/9) + 31/11 sin(2 t + 31/14) + 129/10 sin(3 t + 35/11) + 41/42 sin(4 t + 17/6) + 32/11 sin(5 t + 9/5) + 2/9 sin(6 t + 4/7) + 14/11 sin(7 t + 11/21) + 1/35 sin(8 t + 7/13) - 813/4) θ(7 π - t) θ(t - 3 π) + (-2/3 sin(9/8 - 23 t) - 5/7 sin(1/9 - 20 t) - 4/7 sin(17/11 - 12 t) - 76/9 sin(11/7 - 4 t) - 194/9 sin(1/6 - 3 t) + 16/5 sin(7 t) + 9777/16 sin(t + 22/7) + 234/11 sin(2 t + 9/7) + 7/5 sin(5 t + 77/17) + 59/16 sin(6 t + 85/43) + 4/3 sin(8 t + 121/30) + 7/12 sin(9 t + 5/3) + 5/7 sin(10 t + 6/5) + 19/12 sin(11 t + 13/5) + 16/7 sin(13 t + 1/66) + 5/9 sin(14 t + 43/10) + 8/5 sin(15 t + 32/11) + 40/41 sin(16 t + 9/14) + 51/50 sin(17 t + 4/13) + sin(18 t + 35/8) + 2/3 sin(19 t + 32/9) + 6/11 sin(21 t + 16/5) + 18/19 sin(22 t + 4/5) + 6/7 sin(24 t + 35/11) - 2103/29) θ(3 π - t) θ(t + π)) θ(sqrt(sgn(sin(t/2))))',
            x_range = [-500, 500],
            y_range = [-1000, 500],
            t_ranges = t_ranges,
            **kwargs
        )
        super().__init__(curve, resolution)

    def get_3D_points_on_face(self) -> FacePointData:
        param_func = self.curve.param_func
        t_ranges = self.curve.t_ranges
        
        component_specs = [
            ('outline', [0], 30, 0.7),
            ('hair', [13], 15, 0.9),
            ('ears', [1, 7], 30, 0.6),
            ('eyes', [2, 3, 4, 5, 11, 12], 39, 1),
            ('mouth', [6, 8], 20, 1.2),
            ('nose', [14], 10, 1.45),
            ('extras', [9, 10], 10, 1)
        ]
        
        components = {
            name: _get_component_points(param_func, t_ranges, indices, int(num_points * self.resolution), z_val)
            for name, indices, num_points, z_val in component_specs
        }
        
        return FacePointData(**components)

