import comfy.sd
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]

class SchedulerSelectorKSampler:
    CATEGORY = 'Telegram/utils'
    RETURN_TYPES = (SCHEDULERS,) 
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_names"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (SCHEDULERS,)}}

    def get_names(self, scheduler):
        return (scheduler,)
