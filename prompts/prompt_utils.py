# This file defines several useful functions for working with prompts.

import os
import copy
import logging
import importlib

from typing import Dict, List, Any, Union, Callable

LOGGER = logging.getLogger(__name__)


def clean_prompt_name(prompt_file):
    return prompt_file.split("/")[-1].rstrip(".txt").rstrip(".py")


def load_constants(template_file, constant_name="MESSAGE_TEMPLATE"):
    # This file loads a tconstant from a python file, which helps use dynamically load a new messsage template.
    assert template_file.endswith(".py"), "The template file should be a python file"
    module_name = os.path.splitext(os.path.basename(template_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, template_file)
    if spec is None:
        raise ImportError(f"Cannot import {module_name} from {template_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)[constant_name]


class CustomDefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep the placeholder as-is


def replace_prompt(
    prompt_list: List[Dict[str, str]], replace_dict: Dict[str, str]
) -> List[Dict[str, str]]:
    """This function replaces the placeholders in the prompt list with the values in the replace_dict. For example,
        Hi {{}}, my name is {name} -> Hi {}, my name is John
    if replace_dict = {"name": "John"}.
    It does not replace anything within {{}}.

    Args:
        prompt_list (List[Dict[str, str]]): the message template
        replace_dict (Dict[str, str]): the dictionary containing the values to replace

    Returns:
        List[Dict[str, str]]: the message template with the placeholders replaced
    """
    prompt_list = copy.deepcopy(prompt_list)
    prompt_list = [
        {
            k: v.format_map(CustomDefaultDict(**replace_dict))
            if isinstance(v, str)
            else v
            for k, v in prompt.items()
        }
        for prompt in prompt_list
    ]
    return prompt_list


def build_prompt(
    replace_dicts: Dict[str, str],
    template: Union[List[str], str],
    apply_template_fn: Callable = None,
) -> str:
    """This function builds a prompt by replacing the placeholders in the template with the values in the replace_dicts. And then apply the apply_template_fn to the prompt.
    For example
        [{Hi {{}}, my name is {name}}] -> [Hi {}, my name is John] -> Hi {}, my name is John

    Args:
        replace_dicts (Dict[str, str]): the dictionary containing the values to replace
        template (Union[List[str], str]): the message template
        apply_template_fn (Callable, optional): the function to format the prompt. We assume tokenizer.apply_chate_template. Defaults to None.

    Returns:
        str: the final formated prompt
    """

    if apply_template_fn is None:
        LOGGER.warning(
            "No apply_template_fn is provided. Using default apply_template_fn."
        )
        apply_template_fn = "\n".join if isinstance(template, list) else lambda x: x

    new_prompt = replace_prompt(
        template if isinstance(template, list) else [template], replace_dicts
    )
    prompt = apply_template_fn(new_prompt)
    return prompt
