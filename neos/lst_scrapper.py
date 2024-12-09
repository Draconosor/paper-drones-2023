import os
import re
import pandas as pd

def scrape_lst_file(lst_file_path):
    lst_data = []
    start_marker = 'VARIABLE v.L  1 si se usa el camion k'
    end_marker = 'VARIABLE s.L  1 si se usa el dron l'

    with open(lst_file_path, 'r') as lst_file:
        within_block = False
        block_lines = []

        for line in lst_file:
            if start_marker in line:
                within_block = True
            elif end_marker in line:
                if within_block:
                    lst_data.extend(block_lines)
                    break
            elif within_block:
                block_lines.append(line.strip())

    return lst_data

def get_max_value(lst_data):
    max_value = None

    for line in lst_data:
        values = [float(val.split()[1]) for val in line.split(',') if val.strip()]
        if values:
            current_max = min(values)
            if max_value is None or current_max > max_value:
                max_value = current_max

    return max_value

def count_non_blank_records(lst_data):
    main_list = []
    for line in lst_data:
        values = [float(val.split()[0]) for val in line.split(',') if val.strip()]
        if values:
            main_list += values
    return len(main_list)

def count_depots(lst_data):
    depots = set()
    for line in lst_data:
        match = re.search(r'INDEX\s+(\d+)\s+=\s+Depot\s+(\d+)', line)
        if match:
            depot_number = match.group(2)
            depots.add(depot_number)
    return len(depots)

def extract_third_word_twofourth(filename):
    words = filename.split(' ')
    if len(words) >= 3:
        return f'{words[2]} 0'
    else:
        return None

def get_lst_data(directory):
    lst_df_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.lst'):
                lst_file_path = os.path.join(root, file)
                scraped_data = scrape_lst_file(lst_file_path)
                if scraped_data:
                    #min_value = get_max_value(scraped_data)
                    depots = count_non_blank_records(scraped_data)
                    #if max_value is not None:
                    third_word = extract_third_word_twofourth(file)
                    print(third_word)
                    if third_word is not None:
                        lst_df_data.append({'instance': third_word, 'min_makespan' : depots})

    return pd.DataFrame(lst_df_data)

# Example usage:
directory_path = r'instances/results'
lst_df = get_lst_data(directory_path)
with pd.ExcelWriter('results z2.xlsx', engine='xlsxwriter') as writer:
    lst_df.to_excel(writer, index = False)