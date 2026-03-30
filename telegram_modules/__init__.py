import asyncio
import functools

from .hook import pre_execute, pre_get_input_data
import execution


def _make_async_safe_wrapper(function, prefunction):
    """
    Wrap a function (sync or async) with a prefunction that fires before execution.
    
    If the original function is async (coroutine), returns an async wrapper.
    If the original function is sync, returns a sync wrapper.
    
    This maintains ComfyUI 0.3.68+ compatibility with HierarchicalCache (async)
    while also supporting older versions with sync execution.
    """
    if asyncio.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_run(*args, **kwargs):
            try:
                prefunction(*args, **kwargs)
            except Exception:
                pass
            return await function(*args, **kwargs)
        return async_run
    else:
        @functools.wraps(function)
        def sync_run(*args, **kwargs):
            try:
                prefunction(*args, **kwargs)
            except Exception:
                pass
            return function(*args, **kwargs)
        return sync_run


# Wrap PromptExecutor.execute so pre_execute fires before each generation.
execution.PromptExecutor.execute = _make_async_safe_wrapper(
    execution.PromptExecutor.execute, pre_execute
)

# Wrap get_input_data so pre_get_input_data fires (sets current_save_image_node_id).
if hasattr(execution, "get_input_data"):
    execution.get_input_data = _make_async_safe_wrapper(
        execution.get_input_data, pre_get_input_data
    )
