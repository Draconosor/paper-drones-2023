import os
import zipfile
import re
import pandas as pd

def extract_third_and_fourth_words(filename):
    # Split the filename by spaces and extract the third and fourth words
    words = filename.split(' ')
    if len(words) >= 4:
        return words[2]
    else:
        return None, None

def get_latest_incumbent_lines(directory, pct=False):
    data = {'Instance': [], 'Incumbent Value': [], 'Seconds': [], 'Percentage': []}

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        # Extract each .log file in the zip file
                        for member in zip_ref.namelist():
                            if member.endswith('.log'):
                                with zip_ref.open(member) as log_file:
                                    # Read lines from the log file
                                    lines = log_file.readlines()
                                    incumbent_value_line = None
                                    percentage_line = None
                                    # Search for lines mentioning "Incumbent Value" and percentage
                                    for line in lines:
                                        if b"Found incumbent of value" in line:
                                            incumbent_value_line = line.decode('utf-8')
                                        elif b"%" in line:
                                            percentage_line = line.decode('utf-8')
                                    if incumbent_value_line and percentage_line:
                                        # Extract incumbent value, seconds, and percentage
                                        incumbent_value_match = re.search(r'(\d+\.\d+)', incumbent_value_line)
                                        percentage_match = re.search(r'(\d+\.\d+%|\d+%)', percentage_line)
                                        if incumbent_value_match and percentage_match:
                                            incumbent_value = float(incumbent_value_match.group(1))
                                            seconds = float(re.search(r'after (\d+\.\d+) sec\.', incumbent_value_line).group(1))
                                            percentage = float(re.search(r'(\d+\.\d+|\d+)', percentage_match.group(1)).group(1))
                                            # Add data to the dictionary
                                            data['Instance'].append(f"{extract_third_and_fourth_words(file)} {file.split(' ')[3][:2] if pct else '0'}")
                                            data['Incumbent Value'].append(incumbent_value)
                                            data['Seconds'].append(seconds)
                                            data['Percentage'].append(percentage)
                except zipfile.BadZipFile:
                    with open(zip_file_path[:-3]+'lst', 'r') as log_file:
                        # Read lines from the log file
                        lines = log_file.readlines()
                        incumbent_value_line = None
                        percentage_line = None
                        # Search for lines mentioning "Incumbent Value" and percentage
                        for line in lines:
                            if "Found incumbent of value" in line:
                                incumbent_value_line = line
                            elif "%" in line:
                                percentage_line = line
                        if incumbent_value_line and percentage_line:
                            # Extract incumbent value, seconds, and percentage
                            incumbent_value_match = re.search(r'(\d+\.\d+)', incumbent_value_line)
                            percentage_match = re.search(r'(\d+\.\d+%|\d+%)', percentage_line)
                            if incumbent_value_match and percentage_match:
                                incumbent_value = float(incumbent_value_match.group(1))
                                seconds = float(re.search(r'after (\d+\.\d+) sec\.', incumbent_value_line).group(1))
                                percentage = float(re.search(r'(\d+\.\d+|\d+)', percentage_match.group(1)).group(1))
                                # Add data to the dictionary
                                data['Instance'].append(f"{extract_third_and_fourth_words(file)} {file.split(' ')[3][:2] if pct else '0'}")
                                data['Incumbent Value'].append(incumbent_value)
                                data['Seconds'].append(seconds)
                                data['Percentage'].append(percentage)

    return pd.DataFrame(data)

basic = get_latest_incumbent_lines(r'instances/results')
const = get_latest_incumbent_lines(r'instances/results pcts', pct=True)
complete = pd.concat([basic, const])


def extraer_numeros_grupos(cadena):
    numeros = []
    numero_actual = ''
    for caracter in cadena:
        if caracter.isdigit():
            numero_actual += caracter
        elif numero_actual:
            numeros.append(int(numero_actual))
            numero_actual = ''
    if numero_actual:
        numeros.append(int(numero_actual))
    return tuple(numeros) 

# Aplicar la función para extraer los grupos de números y crear nuevas columnas en el DataFrame
complete['Numeros'] = complete['Instance'].apply(extraer_numeros_grupos)

# Ordenar el DataFrame primero por los grupos de números y luego por el número total
complete = complete.sort_values(by=['Numeros', 'Instance']).reset_index(drop = True)

complete.to_clipboard(index = False)