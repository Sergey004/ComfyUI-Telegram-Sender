import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import piexif
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enum import Enum

import folder_paths

from .. import hook
from ..capture import Capture
from ..trace import Trace
from ..utils.log import print_warning


class SaveImageWithMetaData:
    """
    Minimal version - only metadata extraction methods, no UI/node functionality
    """

    @classmethod
    def gen_pnginfo(cls, prompt, prefer_nearest):
        """
        Extract metadata from workflow
        EXACT copy used by hook system
        """
        inputs = Capture.get_inputs()
        trace_tree_from_this_node = Trace.trace(hook.current_save_image_node_id, prompt)
        inputs_before_this_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_this_node, prefer_nearest)

        sampler_node_id = Trace.find_sampler_node_id(trace_tree_from_this_node)
        if sampler_node_id:
            trace_tree_from_sampler_node = Trace.trace(sampler_node_id, prompt)
            inputs_before_sampler_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_sampler_node, prefer_nearest)
        else:
            inputs_before_sampler_node = {}

        return Capture.gen_pnginfo_dict(inputs_before_sampler_node, inputs_before_this_node, prompt)
