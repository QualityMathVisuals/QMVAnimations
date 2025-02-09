from abc import ABC, abstractmethod
from manimlib import *
import numpy as np
import copy
import hashlib
from scipy import integrate
from scipy.special import jn_zeros, jv
from typing import Callable, List, Tuple, Optional, Union, Type
from QMVlib import *

class Wave1DConfig:
    """Configuration settings for pressure wave properties"""
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, 1),
        resolution: int = 10,
        dampening: float = 0.0,
        wave_speed: float = 2.0,
        colors_for_height: Optional[Tuple[Color, Color]] = None
    ):
        self.u_range = u_range
        self.resolution = resolution
        self.dampening = dampening
        self.wave_speed = wave_speed
        self.colors_for_height = colors_for_height


class Wave1DMobject(TipableVMobject):
    def __init__(self, wave_config: Optional[Wave1DConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.wave_config = wave_config or Wave1DConfig()
        self.init_wave_medium()
        self.wave_functions_to_render = {}
        self.add_updater(lambda m: m.set_points_by_wave_functions())

    def init_wave_medium(self):
        length = self.wave_config.u_range[1] - self.wave_config.u_range[0]
        self.start = LEFT * (length / 2)
        self.end = RIGHT * (length / 2)

    @Mobject.affects_data
    def set_points_by_wave_functions(self):
        self.clear_points()
        x_values = np.linspace(*self.wave_config.u_range, int(self.wave_config.resolution * (self.wave_config.u_range[1] - self.wave_config.u_range[0])))
        y_values = np.array([self.wave_function(x) for x in x_values])
        points = np.column_stack((x_values, y_values, np.zeros_like(x_values)))
        if len(points) > 0:
            self.set_points_smoothly(points)
            self.put_start_and_end_on(self.start, self.end)
            if self.wave_config.colors_for_height:
                color_map = get_colormap_from_colors(self.wave_config.colors_for_height)
                self.data['stroke_rgba'][:] = [color_map([sigmoid(point[1])])[0] for point in self.get_points()]

    def sigmoid(x):
        return 1/(1 + np.exp(-z))

    def add_standing_wave(self, mode, amplitude=1, t_tracker: Optional[ValueTracker] = None, t_initial=0) -> ValueTracker:
        t_tracker = t_tracker if t_tracker is not None else ValueTracker(t_initial)
        def standing_wave(x):
            return self._standing_wave_function(mode, amplitude)(x, t_tracker.get_value())
        self.wave_functions_to_render[mode] = standing_wave
        self.set_points_by_wave_functions()
        self.update()
        return t_tracker

    def remove_standing_wave(self, mode):
        return self.wave_functions_to_render.pop(mode)
    

    def _standing_wave_function(self, mode, amplitude) -> Callable:
        length = (self.wave_config.u_range[1] - self.wave_config.u_range[0])
        k = mode * PI / length
        omega = self.wave_config.wave_speed * k
        beta = self.wave_config.dampening
        omega = np.sqrt(omega ** 2 - beta ** 2)
        def standing_wave_func(x, t):
            return amplitude * np.sin(x * k) * np.cos(omega * t) * np.exp(-beta * t)
        return standing_wave_func

    def wave_function(self, x) -> float:
        components = [wave_func(x) for wave_func in self.wave_functions_to_render.values()]
        return sum(components)


class StandingWave1D(Wave1DMobject):
    def __init__(self, mode, length=1, amplitude=1, dampening=0.0, **kwargs):
        if 'wave_config' in kwargs:
            wave_config = kwargs.pop('wave_config')
        else:
            wave_config = Wave1DConfig()
        wave_config.u_range = (0, length)
        wave_config.dampening = dampening
        super().__init__(wave_config=wave_config, **kwargs)
        self.amplitude = amplitude
        self.t_tracker = self.add_standing_wave(mode, amplitude=amplitude)

    def get_time_tracker(self):
        return self.t_tracker


class FourierWave1D(Wave1DMobject):
    def __init__(self, wave_def: Callable[float, float], num_standing_waves, length=1, dampening=0.0, **kwargs):
        if 'wave_config' in kwargs:
            wave_config = kwargs.pop('wave_config')
        else:
            wave_config = Wave1DConfig()
        wave_config.u_range = (0, length)
        wave_config.dampening = dampening
        super().__init__(wave_config=wave_config, **kwargs)
        self.length = length
        self.t_tracker = ValueTracker(0)
        self.convolution_coeffs = self._get_fourier_coeffs(wave_def, num_standing_waves, length)
        for n, A in enumerate(self.convolution_coeffs):
            self.add_standing_wave(n + 1, A, t_tracker=self.t_tracker)


    def _get_fourier_coeffs(self, function, num_standing_waves, length):
        return [
            (2 / length) * integrate.quad(
                lambda x: function(x) * np.sin((n * PI * x) / length), 
                0, length)[0]
            for n in range(1, num_standing_waves + 1)
        ]

    def get_time_tracker(self):
        return self.t_tracker

    def get_isolated_compnent(self, mode):
        return StandingWave1D(mode, length=self.length, amplitude=self.convolution_coeffs[mode - 1], dampening=self.wave_config.dampening)


class RigidFourierWave1D(FourierWave1D):
    def _get_fourier_coeffs(self, points, num_standing_waves, length) -> List[float]:
        coefficients = []
        for n in range(1, num_standing_waves + 1):
            An = sum(self._segment_contribution(points[i], points[i + 1], n, length)
                    for i in range(len(points) - 1))
            coefficients.append(An)
        return coefficients

    def _segment_contribution(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                            n: int, length: float) -> float:
        x1, y1 = p1
        x2, y2 = p2
        if abs(x1 - x2) < 0.00001:
            return 0
        def indef_integral(x):
            pi_n = PI * n
            dx = x2 - x1
            dy = y2 - y1
            numerator = 2 * (length * dy * np.sin(pi_n * x / length) - 
                           pi_n * (y2 * (x - x1) - y1 * (x - x2)) * 
                           np.cos(pi_n * x / length))
            denominator = PI ** 2 * n ** 2 * dx
            return numerator / denominator
        return indef_integral(x2) - indef_integral(x1)


class StillWave1D(Wave1DMobject):
    def __init__(self, wave_def: Callable[float, float] | List[Tuple[float, float]], length=1, **kwargs):
        if 'wave_config' in kwargs:
            wave_config = kwargs.pop('wave_config')
        else:
            wave_config = Wave1DConfig()
        wave_config.u_range = (0, length)
        super().__init__(wave_config=wave_config, **kwargs)
        function = self._create_wave_function(wave_def)
        self.wave_functions_to_render = {0: function}
        self.clear_updaters()
        self.set_points_by_wave_functions()

    def _create_wave_function(self, y_of_x_or_pts: Union[Callable, List[Tuple[float, float]]]) -> Callable:
        if callable(y_of_x_or_pts):
            return y_of_x_or_pts
        
        if not isinstance(y_of_x_or_pts, list) or not all(isinstance(item, tuple) for item in y_of_x_or_pts):
            raise ValueError("Input must be either a callable or a list of point tuples")
        
        def interpolate_points(x: float) -> float:
            points = y_of_x_or_pts
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x1 <= x <= x2:
                    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            raise ValueError(f"x={x} is out of bounds of the provided points. points[0] = {points[0]}, points[-1] = {points[-1]}")
        
        return interpolate_points


class FourierWave1DBand(FourierWave1D):
    def init_wave_medium(self):
        self.start = 3 * RIGHT
        self.end = 3 * RIGHT

    @Mobject.affects_data
    def set_points_by_wave_functions(self):
        self.clear_points()
        fake_x_values = np.linspace(*self.wave_config.u_range, int(self.wave_config.resolution * (self.wave_config.u_range[1] - self.wave_config.u_range[0])))
        fake_y_values = np.array([self.wave_function(x) for x in fake_x_values])

        theta_values = np.linspace(- 3 * PI / 2, PI / 2, int(self.wave_config.resolution * (self.wave_config.u_range[1] - self.wave_config.u_range[0])))
        x_values = 3 * np.cos(theta_values)
        y_values = 3 * np.sin(theta_values)
        points = np.column_stack((x_values, y_values, fake_y_values))

        if len(points) > 0:
            self.set_points_smoothly(points)
            if self.wave_config.colors_for_height:
                color_map = get_colormap_from_colors(self.wave_config.colors_for_height)
                self.data['stroke_rgba'][:] = [color_map([sigmoid(point[2])])[0] for point in self.get_points()]


class RigidFourierWave1DBand(RigidFourierWave1D):
    def init_wave_medium(self):
        FourierWave1DBand.init_wave_medium(self)

    @Mobject.affects_data
    def set_points_by_wave_functions(self):
        FourierWave1DBand.set_points_by_wave_functions(self)


class Wave1DExampleScene(InteractiveScene):
    def construct(self):
        # Create basic standing wave, then move it around
        standing_wave = StandingWave1D(mode=1, length=10, amplitude=3, color=BLUE_B)
        self.play(ShowCreation(standing_wave, suspend_mobject_updating=True))
        t_tracker = standing_wave.get_time_tracker()
        self.play(
            t_tracker.animate.set_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(standing_wave)

        # Create fourier interpolated wave from function. Move it around
        example_func = lambda x: 1/3 * x * np.sin(x)
        fourier_wave = FourierWave1D(example_func, 10, length=10)
        self.play(ShowCreation(fourier_wave, suspend_mobject_updating=True))
        t_tracker = fourier_wave.get_time_tracker()
        self.play(
            t_tracker.animate.set_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(fourier_wave)

        # Create fourier interpolated wave from point interpolated string. Move it around
        example_points = [
            (0, 0),
            (1, 2),
            (2, 3),
            (3, 0),
            (3, 2),
            (6, 2),
            (6, 0),
            (9, 1),
            (9.99, 3),
            (10, 0),
        ]
        point_fourier_wave = RigidFourierWave1D(example_points, 35, length=10)
        self.play(ShowCreation(point_fourier_wave, suspend_mobject_updating=True))
        t_tracker = point_fourier_wave.get_time_tracker()
        self.play(
            t_tracker.animate.set_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(point_fourier_wave)

        still_wave = StillWave(example_points, length=10)
        self.play(ShowCreation(still_wave))


class Wave2DConfig:
    """Configuration settings for pressure wave properties"""
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, 1),
        v_range: Tuple[float, float] = (0, 1),
        resolution: Tuple[int, int] = (20, 20),
        dampening: float = 0.0,
        wave_speed: float = 2.0,
    ):
        self.u_range = u_range
        self.resolution = resolution
        self.dampening = dampening
        self.wave_speed = wave_speed


class Wave2DMobject(ParametricSurface, ABC):
    def __init__(self, wave_config: Optional[Wave1DConfig] = None, colors_for_height=None, resolution=None, **kwargs):
        self.wave_config = wave_config or Wave1DConfig()
        if resolution is not None:
            self.wave_config.resolution=resolution
        self.x_func, self.y_func = self.init_wave_medium()
        super().__init__(
            lambda u, v: (self.x_func(u, v), self.y_func(u, v), 0),
            u_range=self.wave_config.u_range,
            v_range=self.wave_config.v_range,
            resolution=self.wave_config.resolution,
            **kwargs
        )
        self.wave_functions_to_render = {}
        self.max_amp = 0
        self.add_updater(lambda m: m.set_points_by_wave_functions())
        self.colors_for_height = colors_for_height
        if colors_for_height is not None:
            self.always_update_color_for_height(*colors_for_height)

    @abstractmethod
    def init_wave_medium(self):
        pass

    @abstractmethod
    def _standing_wave_function(self, mode, amplitude, phase) -> Callable:
        pass

    @Mobject.affects_data
    def set_points_by_wave_functions(self):
        self.uv_func = lambda u, v: (self.x_func(u, v), self.y_func(u, v), self.wave_function(u, v))
        super().init_points()

    def add_standing_wave(self, mode, amplitude=1, phase=0, t_tracker: Optional[ValueTracker] = None, t_initial=0) -> ValueTracker:
        t_tracker = t_tracker if t_tracker is not None else ValueTracker(t_initial)
        def standing_wave(u, v):
            return self._standing_wave_function(mode, amplitude, phase)(u, v, t_tracker.get_value())
        self.wave_functions_to_render[mode] = standing_wave
        self.set_points_by_wave_functions()
        if abs(self.max_amp) < amplitude:
            self.max_amp += amplitude
        self.update()
        return t_tracker

    def remove_standing_wave(self, mode):
        self.max_amp -= amplitude
        return self.wave_functions_to_render.pop(mode)

    def wave_function(self, u, v) -> float:
        components = [wave_func(u, v) for wave_func in self.wave_functions_to_render.values()]
        return sum(components)

    def get_mesh(self):
        if self.colors_for_height is None:
            mesh = always_redraw(
                lambda: SurfaceMesh(self, resolution=self.wave_config.resolution).match_style(self)
            )
        else:
            color_map = get_colormap_from_colors(self.colors_for_height)
            color_map_on_points = lambda point: color_map([0.5 + point[2] / self.max_amp])[0]
            def produce_mesh(surf=self):
                mesh = SurfaceMesh(surf, resolution=surf.wave_config.resolution)
                for mob in mesh.get_family(True):
                    stroke_rgba = [[*color_map_on_points(point)] for point in mob.get_points()]
                    stroke_rgba = np.array(stroke_rgba)
                    if len(stroke_rgba) > 0:
                        mob.data['stroke_rgba'][:] = resize_with_interpolation(stroke_rgba, len(mob.data['stroke_rgba']))
                return mesh
            mesh = always_redraw(produce_mesh)
        return mesh

    def always_update_color_for_height(self, *colors):
        color_map = get_colormap_from_colors(colors)
        self.add_updater(
            lambda m: m.set_color_by_rgba_func(lambda point: color_map([0.5 + point[2] / self.max_amp])[0])
        )


class RectangularWave2D(Wave2DMobject, ABC):
    def __init__(self, width: float = 8, height: float = 6, dampening=0.0, **kwargs):
        wave_config = Wave2DConfig()
        wave_config.u_range = (-width / 2, width / 2)
        wave_config.v_range = (-height / 2, height / 2)
        wave_config.dampening = dampening
        self.width = width
        self.height = height
        super().__init__(wave_config=wave_config, **kwargs)

    def init_wave_medium(self):
        x_func = lambda u, v: u
        y_func = lambda u, v: v
        return x_func, y_func
        
    def _standing_wave_function(self, mode, amplitude, phase) -> Callable:
        kx = mode[0] * PI / self.width
        ky = mode[1] * PI / self.height
        omega = self.wave_config.wave_speed * np.sqrt(kx**2 + ky**2)
        beta = self.wave_config.dampening
        omega = np.sqrt(omega ** 2 - beta ** 2)
        def standing_wave_func(u, v, t):
            x_comp = np.sin(kx * u) if mode[0] % 2 == 0 else np.cos(kx * u)
            y_comp = np.sin(ky * v) if mode[1] % 2 == 0 else np.cos(ky * v)
            return amplitude * x_comp * y_comp * np.cos(omega * t) * np.exp(-beta * t)
        return standing_wave_func


class RectangularStandingWave2D(RectangularWave2D):
    def __init__(self, m: int, n: int, amplitude=1, phase=0, **kwargs):
        super().__init__( **kwargs)
        self.mode = (m, n)
        self.amplitude = amplitude
        self.t_tracker = self.add_standing_wave(self.mode, amplitude=amplitude, phase=phase)

    def get_time_tracker(self):
        return self.t_tracker


class RectangularStillWave2D(RectangularWave2D):
    def __init__(self, function: Callable, **kwargs):
        self.function = function
        super().__init__( **kwargs)
        self.clear_updaters()

    def _standing_wave_function(self, mode, amplitude, phase) -> Callable:
        return lambda u, v, t: self.function(u, v)


class RectangularFourierWave2D(RectangularWave2D):
    def __init__(self, function: Callable, num_standing_waves: Tuple[int, int], save_coefficients=True, **kwargs):
        super().__init__(**kwargs)
        self.t_tracker = ValueTracker(0)
        self.convolution_coeffs = self._get_fourier_coeffs(function, self.width, self.height, num_standing_waves, save_coefficients)
        for mode in self.convolution_coeffs.keys():
            self.add_standing_wave(mode, self.convolution_coeffs[mode], t_tracker=self.t_tracker)


    def _get_fourier_coeffs(self, function, width, height, num_standing_waves, save_coefficients):
        uuid=None
        if save_coefficients:
            #Throw a bunch of things in the hash
            u_ins = np.linspace(*self.wave_config.u_range, 25)
            v_ins = np.linspace(*self.wave_config.v_range, 25)
            points_for_hash = [int(function(u, v) * 1000) for u in u_ins for v in v_ins]
            uuid = FourierCalculator.calculate_uuid(
                2, 'rectangular', 
                self.wave_config.u_range,
                self.wave_config.v_range,
                *points_for_hash,
            )            
        calculator = FourierCalculator(function, save_coefficients, uuid=uuid)
        coeffs = {}
        for n in range(num_standing_waves[0]):
            for m in range(num_standing_waves[1]):
                coeffs[(m, n)] = calculator.get_rectangular_fourier_coefficient(m, n, width, height)
        calculator.update_cache()
        return coeffs

    def get_time_tracker(self):
        return self.t_tracker


class RadialWave2D(Wave2DMobject, ABC):
    def __init__(self, radius: float = 5, dampening=0.0, **kwargs):
        wave_config = Wave2DConfig()
        wave_config.u_range = (0, radius)
        wave_config.v_range = (0, TAU)
        wave_config.dampening = dampening
        self.radius = radius
        super().__init__(wave_config=wave_config, **kwargs)

    def init_wave_medium(self):
        x_func = lambda u, v: np.cos(v) * u
        y_func = lambda u, v: np.sin(v) * u
        return x_func, y_func
        
    def _standing_wave_function(self, mode, amplitude, phase) -> Callable:
        n, m = mode
        if n == 0 and m > 0:
            jmn = 0
        elif n > 0 and m >= 0: 
            jmn = jn_zeros(m, n)[-1]
        else:
            raise ValueError(f'Values of m ({m}) and n ({n}) are invalid')
        k = jmn / self.radius
        omega = self.wave_config.wave_speed * k
        beta = self.wave_config.dampening
        omega = np.sqrt(omega ** 2 - beta ** 2)
        def standing_wave_func(u, v, t):
            r_comp = jv(m, k * u)
            thta_comp = np.cos(m * v + phase)
            return amplitude * r_comp * thta_comp * np.cos(omega * t) * np.exp(-beta * t)
        return standing_wave_func


class RadialStandingWave2D(RadialWave2D):
    def __init__(self, m: int, n: int, amplitude=1, phase=0, **kwargs):
        super().__init__( **kwargs)
        self.mode = (m, n)
        self.amplitude = amplitude
        self.t_tracker = self.add_standing_wave(self.mode, amplitude=amplitude, phase=phase)

    def get_time_tracker(self):
        return self.t_tracker


class RadialStillWave2D(RadialWave2D):
    def __init__(self, function: Callable, **kwargs):
        self.function = function
        super().__init__( **kwargs)
        self.clear_updaters()
        self.add_standing_wave((0, 0))

    def _standing_wave_function(self, mode, amplitude, phase) -> Callable:
        return lambda u, v, t: self.function(u, v)


class RadialFourierWave2D(RadialWave2D):
    def __init__(self, function: Callable, num_standing_waves: Tuple[int, int], save_coefficients=True, **kwargs):
        super().__init__(**kwargs)
        self.t_tracker = ValueTracker(0)
        self.convolution_coeffs = self._get_fourier_coeffs(function, radius, num_standing_waves, save_coefficients)
        for mode in self.convolution_coeffs.keys():
            self.add_standing_wave(mode, *self.convolution_coeffs[mode], t_tracker=self.t_tracker)


    def _get_fourier_coeffs(self, function, radius, num_standing_waves, save_coefficients):
        uuid=None
        if save_coefficients:
            #Throw a bunch of things in the hash
            u_ins = np.linspace(*self.wave_config.u_range, 25)
            v_ins = np.linspace(*self.wave_config.v_range, 25)
            points_for_hash = [int(function(u, v) * 1000) for u in u_ins for v in v_ins]
            uuid = FourierCalculator.calculate_uuid(
                2, 'radial', 
                self.wave_config.u_range,
                self.wave_config.v_range,
                *points_for_hash,
            )            
        calculator = FourierCalculator(function, save_coefficients, uuid=uuid)
        coeffs = {}
        for n in range(num_standing_waves[0]):
            for m in range(num_standing_waves[1]):
                coeffs[(m, n)] = calculator.radial_fourier_coefficients(m, n, radius)
        calculator.update_cache()
        return coeffs

    def get_time_tracker(self):
        return self.t_tracker



class FourierCalculator:
    def __init__(self, surface_z, save_coefficients, uuid=None):
        if save_coefficients:
            self.PATH_RELATIVE = 'FourierCoefficientCache.pkl'
            try:
                with open(self.PATH_RELATIVE, 'rb') as f:
                    self.fourier_coeff_cache = pickle.load(f)
            except FileNotFoundError:
                self.fourier_coeff_cache = {uuid: {}}
            if uuid not in self.fourier_coeff_cache:
                self.fourier_coeff_cache[uuid] = {}
            self.uuid = uuid
        else:
            self.fourier_coeff_cache = {0: {}}
            self.uuid = 0
        self.save_coefficients = save_coefficients
        self.surface_z = surface_z
        self.should_update_cache = False

    @staticmethod
    def calculate_uuid(dimension:int, membrane_type:str, *unique_args):
        sha = hashlib.sha224()
        sha.update(str(dimension).encode('ascii'))
        sha.update(membrane_type.encode('ascii'))
        for arg in unique_args:
            sha.update(str(arg).encode('ascii'))
        uuid = sha.hexdigest()
        return uuid

    @staticmethod
    def convolution_radial_mode(f, m, k_mn, R):
        if m == 0:
            # When there is radial symmetry, no need for phase
            S_mn, _ = integrate.dblquad(
                lambda r, theta: (jv(m, k_mn * r))**2 * r,
                0, TAU,
                lambda _: 0, lambda _: R
            )
            A_mn, _ = integrate.dblquad(
                lambda r, theta: f(r, theta) * jv(m, k_mn * r) * r,
                0, TAU,
                lambda _: 0, lambda _: R
            )
            return A_mn / S_mn, 0
        
        # Compute normalization constants
        S_mn, _ = integrate.dblquad(
            lambda r, theta: (jv(m, k_mn * r) * np.cos(m * theta))**2 * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        T_mn, _ = integrate.dblquad(
            lambda r, theta: (jv(m, k_mn * r) * np.sin(m * theta))**2 * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        
        # Compute the coefficients A_mn, B_mn
        A_mn, _ = integrate.dblquad(
            lambda r, theta: f(r, theta) * jv(m, k_mn * r) * np.cos(m * theta) * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        B_mn, _ = integrate.dblquad(
            lambda r, theta: f(r, theta) * jv(m, k_mn * r) * np.sin(m * theta) * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        # Normalize
        A_mn = A_mn / S_mn
        B_mn = B_mn / T_mn

        #Amplitude and phase
        C_mn = np.sqrt(A_mn ** 2 + B_mn ** 2)
        phi_mn = np.arctan2(-B_mn, A_mn)
        return C_mn, phi_mn

    @staticmethod
    def convolution_rectangular_mode(f, m, n, width, height, k_x, k_y):
        # Compute normalization constants
        S_mn = (width * height) / 4
        def phimn(u, v):
            y_comp = np.sin(k_y * v) if n % 2 == 0 else np.cos(k_y * v)
            x_comp = np.sin(k_x * u) if m % 2 == 0 else np.cos(k_x * u)
            return x_comp * y_comp
        
        # Compute the coefficients A_mn, B_mn

        A_mn, acc = integrate.dblquad(
            lambda x, y: f(x, y) * phimn(x, y),
            -height/2, height/2,
            lambda _: -width/2, lambda _: width/2
        )
        return A_mn / S_mn


    def radial_fourier_coefficients(self, m: int, n: int, R):
        if (m, n) not in self.fourier_coeff_cache[self.uuid]:
            print(f'Calculating new FC: {(m, n, R)}')
            jmn = jn_zeros(m, n)[-1]
            kmn = jmn / R
            self.fourier_coeff_cache[self.uuid][(m, n)] = self.convolution_radial_mode(self.surface_z, m, kmn, R)
            self.should_update_cache = True
        return self.fourier_coeff_cache[self.uuid][(m, n)]

    def get_rectangular_fourier_coefficient(self, m: int, n: int, width: float, height: float):
        if (m, n) not in self.fourier_coeff_cache[self.uuid]:
            print(f'Calculating new FC: {(m, n, width, height)}')
            kx = PI * m / width
            ky = PI * n / height
            self.fourier_coeff_cache[self.uuid][(m, n)] = self.convolution_rectangular_mode(self.surface_z, m, n, width, height, kx, ky)
            self.should_update_cache = True
        return self.fourier_coeff_cache[self.uuid][(m, n)]

    def update_cache(self):
        if self.should_update_cache and self.save_coefficients:
            with open(self.PATH_RELATIVE, 'wb+') as f:
                pickle.dump(self.fourier_coeff_cache, f)
            self.should_update_cache = False


class Radial2DExampleScene(InteractiveScene):
    def construct(self):
        #Standing Wave
        standing_wave = RadialStandingWave2D(2, 1, amplitude=3, colors_for_height=[QMV_BLUE_C, GREY_B, QMV_PINK_C])
        self.play(ShowCreation(standing_wave, suspend_mobject_updating=True))
        t_tracker = standing_wave.get_time_tracker()
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(standing_wave)

        #Standing Wave Mesh
        standing_wave_mesh = standing_wave.get_mesh()
        self.play(ShowCreation(standing_wave_mesh, suspend_mobject_updating=True))
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(standing_wave_mesh)

        #Face Wave
        radius = 4
        pt_to_surface_z = ObamaFace(max_dist_from_center=radius).get_3D_param_radial(radius)
        still_wave = RadialStillWave2D(pt_to_surface_z, resolution=(50, 50))
        self.play(ShowCreation(still_wave))
        self.remove(still_wave)

        # Fourier Wave
        fourier_wave = RadialFourierWave2D(pts_to_surface_z, (5, 5), colors_for_height=[BLUE, GREY_B, PINK])
        self.play(ShowCreation(fourier_wave, suspend_mobject_updating=True))
        t_tracker = fourier_wave.get_time_tracker()
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )



class Rectangular2DExampleScene(InteractiveScene):
    def construct(self):
        #Standing Wave
        standing_wave = RectangularStandingWave2D(2, 1, amplitude=3, colors_for_height=[QMV_BLUE_C, GREY_B, QMV_PINK_C])
        self.play(ShowCreation(standing_wave, suspend_mobject_updating=True))
        t_tracker = standing_wave.get_time_tracker()
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(standing_wave)

        #Standing Wave Mesh
        standing_wave_mesh = standing_wave.get_mesh()
        self.play(ShowCreation(standing_wave_mesh, suspend_mobject_updating=True))
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(standing_wave_mesh)

        #Face Wave
        width, height = 8, 6
        pts_to_surface_z = ObamaFace().get_3D_param_rectangular(width, height)
        still_wave = RectangularStillWave2D(pts_to_surface_z, resolution=(50, 50))
        self.play(ShowCreation(still_wave))
        self.remove(still_wave)

        # Fourier Wave
        fourier_wave = RectangularFourierWave2D(pts_to_surface_z, (5, 5), colors_for_height=[BLUE, GREY_B, PINK])
        self.play(ShowCreation(fourier_wave, suspend_mobject_updating=True))
        t_tracker = fourier_wave.get_time_tracker()
        self.play(
            t_tracker.animate.increment_value(10),
            run_time=10,
            rate_func=linear
        )



