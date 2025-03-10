import os
import importlib.util


def load_function(template_file, function_name="filter_fn") -> callable:
    """
    Dynamically loads a function from a given Python file.

    :param template_file: Path to the Python file containing the function.
    :param function_name: Name of the function to load.
    :return: The function object if found, otherwise raises an AttributeError.
    """
    assert template_file.endswith(".py"), "The template file should be a Python file"

    module_name = os.path.splitext(os.path.basename(template_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, template_file)
    if spec is None:
        raise ImportError(f"Cannot import {module_name} from {template_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"The function '{function_name}' was not found in {template_file}"
        )

    return getattr(module, function_name)
