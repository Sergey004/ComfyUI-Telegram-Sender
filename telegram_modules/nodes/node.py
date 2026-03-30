"""
Minimal SaveImageWithMetaData node for ComfyUI 0.3.68+
Async wrapper for ComfyUI integration

All actual functionality is delegated to:
- telegram_modules.capture.Capture for metadata extraction
- telegram_sender.py for Telegram integration

This node simply handles the ComfyUI async interface.
"""

import json
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .. import hook
from ..capture import Capture
from ..trace import Trace
from ..utils.log import print_warning


class SaveImageWithMetaData:
    """
    Minimal SaveImageWithMetaData node
    
    PURPOSE: Handle ComfyUI async interface for ComfyUI 0.3.68+
    
    DELEGATED FUNCTIONALITY:
    - Metadata capture: telegram_modules.capture.Capture (async)
    - Telegram sending: telegram_sender.TelegramSender (async)
    - UI: telegram_sender.TelegramSender (has full INPUT_TYPES)
    
    This node is intentionally minimal - it's a compatibility shim.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """Define UI input types (minimal)"""
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save"}),
            },
            "optional": {},
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Metadata extraction node (async, ComfyUI 0.3.68+)"

    @classmethod
    async def gen_pnginfo(cls, prompt, prefer_nearest=True, batch_index=0):
        """
        Extract metadata from workflow (async version for ComfyUI 0.3.68+)
        """
        # Await the async Capture.get_inputs() call
        inputs = await Capture.get_inputs()
        
        trace_tree_from_this_node = Trace.trace(hook.current_save_image_node_id, prompt)
        inputs_before_this_node = Trace.filter_inputs_by_trace_tree(
            inputs, trace_tree_from_this_node, prefer_nearest
        )

        sampler_node_id = Trace.find_sampler_node_id(trace_tree_from_this_node)
        if sampler_node_id:
            trace_tree_from_sampler_node = Trace.trace(sampler_node_id, prompt)
            inputs_before_sampler_node = Trace.filter_inputs_by_trace_tree(
                inputs, trace_tree_from_sampler_node, prefer_nearest
            )
        else:
            inputs_before_sampler_node = {}

        return Capture.gen_pnginfo_dict(
            inputs_before_sampler_node, inputs_before_this_node, prompt, 
            batch_index=batch_index
        )

    async def execute(self, images, prompt=None, extra_pnginfo=None):
        """
        Async execute method for ComfyUI 0.3.68+
        
        Simply returns images unchanged.
        Actual work (Telegram sending, metadata attachment) is handled by:
        - telegram_sender.TelegramSender node (registered separately)
        - The hook system in telegram_modules
        """
        # This node is primarily for registration and interface compatibility
        # The actual workflow sends through TelegramSender node
        return (images,)

