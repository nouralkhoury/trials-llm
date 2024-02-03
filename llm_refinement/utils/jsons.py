import json


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")


def dump_json(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        return data
    except TypeError as e:
        print(f"Unable to serialize the object: {e}")


def loads_json(json_str):
    try:
        return json.loads(json_str)
    except TypeError as e:
        print(f"Unable to load JSON from string: {e}")


def flatten_lists_in_dict(input_dict):
    """
    Flatten lists in a dictionary.

    Parameters:
    - input_dict (dict): Input dictionary containing lists.

    Returns:
    - dict: Output dictionary with flattened lists.

    Example:
    - Input: {"inclusion": [["A", "B"], ["C"]], "exclusion": [['k']]}
    - Output: {"inclusion": ["A", "B", "C"], "exclusion": ['k']}
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            flattened_list = [item for sublist in value for item in (sublist if isinstance(sublist, list) else [sublist])]
            output_dict[key] = flattened_list
        else:
            output_dict[key] = value
    return output_dict
