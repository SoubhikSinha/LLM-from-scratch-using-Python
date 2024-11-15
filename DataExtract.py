import lzma # For handling (.xz) files
from tqdm import tqdm # Progess bar
import os

def xz_files_in_dir(directory): # Input : Directory , output : list of all the (.xz) files
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


folder_path = "D:/Datasets/OpenWebText2 - Copy" # Where the (.xz) files are located
output_file_train = "output_train.txt" # Pattern for output (training) file name
output_file_val = "output_val.txt" # Pattern for output (validation) file name
vocab_file = "vocab.txt" # File for saving the vocabulary

# split_files = int(input("How many files would you like to split this into ?")) # How many splits

files = xz_files_in_dir(folder_path)
total_files = len(files)

# max_count = total_files // split_files if split_files != 0 else total_files # Comparing and setting max. count to total files or split files

# Calculating the split indices
split_index = int(total_files * 0.9) # 90% training
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set() # Collection of unique words - just like a vocabulary / dictionary of words


# Now we need to extract the content of each compressed (.xz) file,
# Read its content and put all the content of that file into a a new output file

# for i in range(split_files):
#     with open(output_file.format(i), "w", encoding="utf-8") as outfile:
#         for count, filename in enumerate(tqdm(files[:max_count], total = max_count)):
#             if count >= max_count:
#                 break
#             file_path = os.path.join(folder_path, filename)
#             with lzma.open(file_path, "rt", encoding="utf-8") as infile:
#                 text = infile.read()
#                 outfile.write(text)
#                 characters = set(text)
#                 vocab.update(characters)
#         files = files[max_count:]

# Processing Training Files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Processing Validation Files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Writing vocabulary into the vocab file
with open(vocab_file, "w", encoding = "utf-8") as vfile:
    for char in vocab:
        vfile.write(char + "\n")


