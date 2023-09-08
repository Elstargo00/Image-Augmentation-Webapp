def text_to_hex(text):
    # encode the text to bytes, then convert to hexadecimal
    hex_output = text.encode("utf-8").hex()
    return hex_output


def extract_transforming_name(obj):
    # Get the full class name
    full_name = str(obj)
    # Extract the class name
    class_name = full_name.split('.')[-1].replace("'>", '')
    return class_name