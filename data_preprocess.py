data1_path = '../data/道德经全文-utf8.txt'
data2_path = '../data/唐诗三百首.txt'
data3_path = '../data/庄子.txt'
data4_path = '../data/边城.txt'
data5_path = '../data/诗经-utf8.txt'

# Function to read file with detected encoding, if it's not utf-8, convert it into utf-8
def convert_files_into_utf8(files_path):
    for file_path in files_path:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            print(f"Detected encoding: {encoding}")

        with open(file_path, 'r', encoding=encoding, errors='replace') as fread:
            if encoding != 'utf-8':
                context = fread.read()
                output_file_path = ''.join(file_path.split('.txt')[0]) + '-utf8.txt'
                with open(output_file_path, 'w', encoding='utf-8') as fwrite:
                    fwrite.write(context)
        if encoding != 'utf-8':
            print (f'convert it into utf-8 successfully')

import os

# List of file names to merge
file_list = [data1_path, data2_path, data3_path, data4_path, data5_path]
output_file = '../data/merged_file.txt'

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    for fname in file_list:
        # Ensure the file exists before trying to read it
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf-8') as infile:
                # Read the contents of the file and write them to the output file
                content = infile.read()
                outfile.write(content)
                # Optionally, add a newline between files
                outfile.write('\n')
        else:
            print(f"File {fname} does not exist.")

print(f"All files have been merged into {output_file}")



output_file = '../data/songs_file.txt'

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('../data/songs_data/'):
        for file in files:
            fname = os.path.join(root, file)
            song_name = file.split('-')[1].split('.')[0]
            # for fname in file_list:
            # Ensure the file exists before trying to read it
            outfile.write(f'歌名 {song_name} \n')
            if os.path.isfile(fname):
                with open(fname, 'r', encoding='utf-8') as infile:
                    # Read the contents of the file and write them to the output file
                    content = infile.read()
                    outfile.write(content)
                    # Optionally, add a newline between files
                    outfile.write('\n')
                    outfile.write('\n')
            else:
                print(f"File {fname} does not exist.")

print(f"All files have been merged into {output_file}")


# for root, dirs, files in os.walk('../data/songs_data/'):
#     for file in files:
#         print (os.path.join(root, file))

import os

# List of file names to merge
data1_path = '../data/songs_file.txt'
data2_path = '../data/merged_file.txt'
file_list = [data1_path, data2_path]
output_file = '../data/songs_poetry_merged_file.txt'

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    for fname in file_list:
        # Ensure the file exists before trying to read it
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf-8') as infile:
                # Read the contents of the file and write them to the output file
                content = infile.read()
                outfile.write(content)
                # Optionally, add a newline between files
                outfile.write('\n')
        else:
            print(f"File {fname} does not exist.")

print(f"All files have been merged into {output_file}")



