"""
Metadata extraction for TelegramSender
EXACT copy of SaveImageWithMetaData.gen_pnginfo() from comfyui_image_metadata_extension
"""

try:
    from .telegram_modules import hook
    from .telegram_modules.capture import Capture
    from .telegram_modules.trace import Trace
    CAPTURE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CAPTURE_AVAILABLE = False


class TelegramMetadata:
    """
    Direct copy of gen_pnginfo from SaveImageWithMetaData
    This ensures 100% compatibility with the original extension
    """
    
    @classmethod
    def gen_pnginfo(cls, prompt, prefer_nearest=True):
        """
        EXACT copy of SaveImageWithMetaData.gen_pnginfo()
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
    
    @staticmethod
    def get_metadata(prompt, prefer_nearest=True):
        """
        Extract metadata from workflow using same logic as SaveImageWithMetaData
        """
        if not CAPTURE_AVAILABLE:
            return {}
        
        try:
            return TelegramMetadata.gen_pnginfo(prompt, prefer_nearest)
        except Exception as e:
            print(f"[TelegramMetadata] Warning: Could not extract metadata: {e}")
            return {}
    
    @staticmethod
    def get_parameters_str(pnginfo_dict):
        """
        Get A1111-style parameters string
        """
        if not CAPTURE_AVAILABLE:
            return ""
        
        try:
            return Capture.gen_parameters_str(pnginfo_dict)
        except Exception as e:
            print(f"[TelegramMetadata] Warning: Could not format parameters: {e}")
            return ""
