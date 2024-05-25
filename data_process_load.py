import os
import yaml
import torch
import chardet
from tokenizer_fun import tokenizer_char

    
with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
    config = yaml.safe_load(yaml_file)

seq_len = config['seq_len']
batch_size = config['batch_size']
device = config['device']
config_vocab_size = config['vocab_size']

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


def merge_files(input_files_list, output_file='../data/merged_file.txt'):
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in input_files_list:
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

    
# read original text data
# data_path = 'data/input.txt'
data_path = 'data/merged_file.txt'
def get_dataloader(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    encode, decode, vocab_size = tokenizer_char(text=text)
    assert vocab_size == config_vocab_size
    # convert data into array
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    test_data = data[n:]

    # build a dataloader
    def get_batch(split='train'):
        # generate a mini batch of data of x and y
        data = train_data if split == 'train' else test_data
        idx = torch.randint(len(data) - seq_len , (batch_size, ))
        x = torch.stack([data[i: i+seq_len] for i in idx])
        y = torch.stack([data[i+1: i+seq_len+1] for i in idx])
        x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, encode, decode