from .nodes.node import SaveImageWithMetaData

current_prompt = {}
current_extra_data = {}
prompt_executer = None
current_save_image_node_id = -1


def pre_execute(self, prompt, prompt_id, extra_data, execute_outputs):
    global current_prompt
    global current_extra_data
    global prompt_executer

    current_prompt = prompt
    current_extra_data = extra_data
    prompt_executer = self


def pre_get_input_data(inputs, class_def, unique_id, *args):
    global current_save_image_node_id

    # Check by class name to avoid circular imports
    class_name = class_def.__name__ if hasattr(class_def, '__name__') else str(class_def)
    
    if class_def == SaveImageWithMetaData or class_name == "TelegramSender":
        current_save_image_node_id = unique_id
