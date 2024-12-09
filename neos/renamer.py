import os

def remove_last_two_characters(filename):
    parts = filename.split(' ')
    if len(parts) > 1:
        parts[2] = parts[2][:-2]  # Remove last two characters from the first word
        return ' '.join(parts)
    return filename

def main():
    folder_path = 'instances/results pcts'  # Replace this with the path to your folder
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        new_filename = remove_last_two_characters(filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed {filename} to {new_filename}')

if __name__ == "__main__":
    main()
