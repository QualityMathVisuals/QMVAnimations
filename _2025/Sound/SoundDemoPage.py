
from manimlib import *
import copy
import hashlib
from QMVlib import *



class StringEncirclePreview(InteractiveScene):
    def construct(self):
        # Create fourier interpolated wave from function. Move it around
        wave_config = Wave1DConfig(colors_for_height=[QMV_BLUE_C, WHITE, QMV_PINK_C])
        example_func = lambda x: 1/3 * x * np.sin(x)
        fourier_wave = FourierWave1D(example_func, 10, length=10, wave_config=wave_config)
        self.play(ShowCreation(fourier_wave, suspend_mobject_updating=True))
        t_tracker = fourier_wave.get_time_tracker()
        self.play(t_tracker.animate.set_value(3), run_time=3, rate_func=there_and_back)
        fourier_wave.suspend_updating()
        
        # Transform to band
        self.play(self.frame.animate.set_phi(1).scale(1.2), run_time=0.5)
        self.frame.add_ambient_rotation(3 * DEGREES)
        self.play(fourier_wave.animate.rotate(90 * DEGREES, RIGHT, ORIGIN), run_time=0.5)
        self.play(fourier_wave.animate.shift(DOWN * 3), run_time=0.5)
        fourier_band = FourierWave1DBand(example_func, 10, length=10, wave_config=wave_config)
        fourier_band.suspend_updating()
        self.play(ReplacementTransform(fourier_wave, fourier_band), run_time=1)
        self.add(Sphere(radius=0.1, color=RED).move_to(UP * 3))

        # Play band moving
        t_tracker = fourier_band.get_time_tracker()
        fourier_band.resume_updating()
        self.play(t_tracker.animate.set_value(10), run_time=10, rate_func=linear)
        self.remove(fourier_band)

        # Rigid band
        # example_points = [
        #     (0, 0),
        #     (1, 2),
        #     (2, 3),
        #     (3, 0),
        #     (3, 2),
        #     (6, 2),
        #     (6, 0),
        #     (9, 1),
        #     (9.99, 3),
        #     (10, 0),
        # ]
        # point_fourier_wave = RigidFourierWave1DBand(example_points, 15, length=10, wave_config=wave_config)
        # self.play(ShowCreation(point_fourier_wave, suspend_mobject_updating=True))
        # t_tracker = point_fourier_wave.get_time_tracker()
        # self.play(
        #     t_tracker.animate.set_value(10),
        #     run_time=10,
        #     rate_func=linear
        # )
        # self.remove(point_fourier_wave)

