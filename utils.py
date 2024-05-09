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


def date_range(start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.strptime(end, '%Y-%m-%d').date()
    delta = end_date - start_date
    days = [start_date + timedelta(days=i) for i in
            range(delta.days + 1)]
    return list(map(lambda n: n.strftime("%Y-%m-%d"), days))