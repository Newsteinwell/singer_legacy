import os
import torch
import torch.nn as nn
import yaml
from torch.nn import functional as F
from data_process_load import get_dataloader
from datetime import datetime
from utils import print_model_params_num 


with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
    config = yaml.safe_load(yaml_file)

total_training_steps = config['total_training_steps']
eval_interval = config['eval_interval']
eval_iters = config['eval_iters']
device = config['device']

data_path = 'data/merged_file.txt'
get_batch, encode, decode = get_dataloader(data_path=data_path)

@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model):
    # create a PyTorch optimizer
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for steps in range(total_training_steps+1):
        if steps % eval_interval == 0 :
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"step: {steps}, train loss: {losses['train']:.4f}, eval loss: {losses['test']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

# save and load model
# Get the current date and time
def save_model(model, prefix='./ckpts'):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = data_path.split('/')[-1].split('.')[0]
    model_filename = os.path.join(prefix, f'model_{dataset_name}_{current_time}.pth')
    torch.save(model.state_dict(), model_filename)
    print (f'****** Save Model : {model_filename} Successfully ! ******')
    return model_filename

def generate_text(model, max_new_tokens=300, save_output=True):
    model.eval()
    generated_idx = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=max_new_tokens)
    generated_text = decode(generated_idx[0].tolist())
    print (generated_text)
    if save_output:
        save_text(generated_text, 'output.txt') 
    model.train()

def save_text(text, filename):
    count = 1
    while True:
        try:
            with open(filename, "x"):
                # If the file does not exist, create it and write to it
                with open(filename, "w") as file:
                    file.write(text)
            break  # Break the loop if successful
        except FileExistsError:
            # If the file exists, try with a different filename
            filename = f"{filename.split('.')[0]}_{count}.txt"
            count += 1

def inference(Model=None, model=None, model_filename=None,  input_text='请给我写一首诗', save_output=True, max_new_tokens=1000):
    if model is None:
        model = Model()
        model.load_state_dict(torch.load(model_filename))
        print (f'load model: {model_filename} successfully!')
        print_model_params_num(model)
    model.to(device)
    model.eval()
    input_token_ids = encode(input_text)
    input_texts = decode(input_token_ids).replace(' ', '*')
    print (f'input text is {input_text}, input of model is {input_texts}')
    generated_idx = model.generate(idx=torch.tensor(input_token_ids, dtype=torch.long, device=device).view(1, -1), max_new_tokens=max_new_tokens, decode=None)
    generated_text = decode(generated_idx[0].tolist())
    print ()
    print (generated_text)
    if save_output:
        save_text(generated_text, './results/infer.txt')    
