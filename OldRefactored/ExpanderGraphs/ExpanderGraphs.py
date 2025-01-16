from abc import ABC, abstractmethod
from manimlib import *
import numpy as np
import copy
import hashlib
from scipy import integrate
from typing import Callable, List, Tuple, Optional, Union, Type
import sys


class GraphConfig:
    """Configuration settings for Expander Graphs"""
    def __init__(
        self,
        vertex_kwargs: List[dict] = None, # Dictionary for each element in the vgroup
        edge_kwargs: List[dict] = None, # Dictionary for each element in the vgroup
        bounding_box_padding = 1.,
        edge_length_mean = 1.,
        edge_length_stdev = 1.,
        theta_spacing_mean = 1.,
        theta_spacing_stdev = 1.,
        theta_center_stdev = 1., #The theta_center_mean is away from any neighbor by default
    ):
        if vertex_kwargs is None:
            vertex_kwargs = [
                {
                    "radius": self.vertex_radius,
                    "color": vertex_color,
                    "fill_opacity": 0.4
                },
                {}
            ]

        if edge_kwargs is None:
            edge_kwargs = [
                {}
            ]
        self.vertex_kwargs = vertex_kwargs
        self.edge_kwargs = edge_kwargs
        self.bounding_box_padding = bounding_box_padding
        self.edge_length_mean = edge_length_mean
        self.edge_length_stdev = edge_length_stdev
        self.theta_spacing_mean = theta_spacing_mean
        self.theta_spacing_stdev = theta_spacing_stdev
        self.theta_center_stdev = theta_center_stdev


class SSGraphConfig(GraphConfig):
    """Configuration settings for SSGraphs"""
    def __init__(
        self,
        **expander_kwargs
    ):
    super().__init__(**expander_kwargs)

class SSGraphWeb(LeveledExpanderGraph, SSGraph):
    pass

class SSGraph(ExpanderGraph, ABC):
    def __init__(self, p: int, ell: int, ss_graph_config: SSGraphConfig, **kwargs):
        self.p = p
        self.ell = ell
        super().__init__(**kwargs)

    def get_prime_label():
        pass

    def get_ell_label():
        pass



class LeveledExpanderGraph(ExpanderGraph, ABC):
    '''Adds a level structure to the leveled expander graph. With an origin specified'''
    def __init__(self, initial_vertex_id, level_colors: Callable[int, ManimColor] = None, level_select_colors: Callable[int, ManimColor] = None, **kwargs):
        super().__init__(initial_vertex_id)
        self.vertex_levels = {initial_vertex_id: 0}
        self.vertices[initial_vertex_id] = self._make_new_vertex(initial_vertex_id)
        self.level_colors = level_colors
        self.level_select_colors = level_select_colors

    # Should only return newly created neighbor ids.
    def _make_new_neighbors(vertex_id):
        curr_level = self.vertex_levels[vertex_id]
        new_neighbor_ids = super()._make_new_neighbors(vertex_id) # At this point they should be positioned correectly
        for new_vert_id in new_neighbor_ids:
            self.vertex_levels[new_vert_id] = curr_level + 1
            # Set custom color based on level colors function, if exits
            if self.level_colors is not None:
                self.vertices[new_vert_id].set_color(self.level_colors(curr_level + 1))
        return new_neighbor_ids

    def get_select_color_vertex(vertex_id):
        if self.level_select_colors is None:
            return super().get_select_color_vertex(vertex_id)
        else:
            return self.level_select_colors(self.vertex_levels[vertex_id])





class SSBFSExample(InteractiveScene):
    def construct(self):
        ss_web = SSGraphWeb()