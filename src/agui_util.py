from typing import Any
from ag_ui.core.events import Event


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case string to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])


def convert_dict_to_camel_case(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert dictionary keys from snake_case to camelCase."""
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        camel_key = to_camel_case(key)
        
        if isinstance(value, dict):
            result[camel_key] = convert_dict_to_camel_case(value)
        elif isinstance(value, list):
            result[camel_key] = [
                convert_dict_to_camel_case(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[camel_key] = value
    
    return result


def encode_event(event: Event) -> dict[str, Any]:
    """
    Convert an Event object to a dictionary with camelCase properties 
    suitable for sending to a JavaScript server.
    """
    # Convert the event to a dictionary using Pydantic's model_dump
    event_dict = event.model_dump()
    
    # Convert all keys to camelCase
    camel_case_dict = convert_dict_to_camel_case(event_dict)
    
    return camel_case_dict
    