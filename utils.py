import re
from datetime import datetime, timedelta


def extract_values(output):
    # Regular expression pattern to match the float value
    pattern = r'array\((.*?)\)'
    extracted_values = []
    for x in output:
        if 'xarray' not in x:
            extracted_values.append(float(x))
        else:
            # If x is not already a float, extract the float value using regular expression
            float_value = re.search(pattern, str(x)).group(1)
            extracted_values.append(float(float_value))

    return extracted_values

