import os
import torch
import torch.distributed
import torch.nn as nn
import yaml
import deepspeed
import json
from torch.nn import functional as F
from torch.optim import AdamW
import torch.optim as optim
from data_process_load import get_dataloader, get_dataloader_ds
from datetime import datetime
from utils import print_model_params_num 

with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
    config = yaml.safe_load(yaml_file)

with open('ds_config.json', 'r') as file:
    ds_config = json.load(file)

total_training_steps = config['total_training_steps']
eval_interval = config['eval_interval']
eval_iters = config['eval_iters']
device = config['device']
epochs = config['epochs']
lr = config['learning_rate']
save_per_step = config['save_per_step']

micro_batch_size = ds_config['train_micro_batch_size_per_gpu']

# data_path = 'data/merged_file.txt'
data_path = 'data/songs_poetry_merged_file.txt'
get_batch, encode, decode = get_dataloader(data_path=data_path)
tr_dataloader, ts_dataloader, encode, decode = get_dataloader_ds(data_path=data_path, batch_size=micro_batch_size)

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

@torch.no_grad()
def estimate_loss_ds(model, dataloader):
    model.eval()
    losses = torch.zeros(len(dataloader))
    steps = 0
    for batch in dataloader:
        if steps % 1000 == 0:
            print (f'eval_step -- : {steps}')
        X, Y = batch
        X, Y = X.cuda(), Y.cuda() 
        logits, loss = model(X, Y)
        losses[steps] = loss.item()
        out = losses.mean()
        steps += 1
    model.train()
    return out

def train_model(model):
    # create a PyTorch optimizer
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for steps in range(total_training_steps+1):
        if steps % eval_interval == 0 :
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"step: {steps}, train loss: {losses['train']:.4f}, eval loss: {losses['test']:.4f}")
        xb, yb = get_batch('train')
        # print (f'xb.shape: {xb.shape}, yb.shape: {yb.shape}')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

def train_model_ds(model):
    # create a PyTorch optimizer
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.8)
    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                             model_parameters=model.parameters(),
                             config = ds_config,
                             optimizer=optimizer)
                            #  lr_scheduler=scheduler)
    # model.to(device)
    for epoch in range(epochs):
        print (f'Epoch ---: {epoch}')
        model.train()
        # for steps in range(total_training_steps+1):
        steps = 0
        print (f'Train data length {len(tr_dataloader)}, Test data length: {len(ts_dataloader)}')
        for batch in tr_dataloader:
            xb, yb = batch
            xb, yb = xb.cuda(), yb.cuda()
            # print (f'xb.shape: {xb.shape}, yb.shape: {yb.shape}')
            if steps % eval_interval == 0 :
                rank = torch.distributed.get_rank()
                # tr_loss = estimate_loss_ds(model, tr_dataloader)
                # ts_loss = estimate_loss_ds(model, ts_dataloader)
                lr_scheduler = optimizer.param_groups[0]['lr']
                # print(f"rank: {rank}, step: {steps}, LR: {lr_scheduler}, train loss: {'Not test'}, eval loss: {ts_loss:.4f}")
                print(f"rank: {rank}, step: {steps}, LR: {lr_scheduler}, train loss: {'Not test'}, eval loss: {'Not test'}")
            model.zero_grad()
            logits, loss = model(xb, yb)
            model.backward(loss)
            model.step()
            # scheduler.step()

            if steps % save_per_step == 0:
                save_model_ds(model, prefix='./ckpts_ds', suffix=f'epoch_{epoch}_step_{steps}')
            steps+=1
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

# Get the current date and time
def save_model_ds(model, prefix='./ckpts_ds', suffix=None):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = data_path.split('/')[-1].split('.')[0]
    if suffix is not None:
        model_filename = os.path.join(prefix, f'model_{dataset_name}_{suffix}_{current_time}.pth')
    else:
        model_filename = os.path.join(prefix, f'model_{dataset_name}_{current_time}.pth')
    model.save_checkpoint(model_filename)
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
    filename_save = filename
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
            filename = f"{filename_save.split('.')[0]}_{count}.txt"
            count += 1

def inference_ds(Model=None, model=None, model_filename=None,  input_text='请给我写一首诗', save_output=True, max_new_tokens=1000):
    if model is None:
        model = Model()
        optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
        model, _, _, _ = deepspeed.initialize(model=model,
                             model_parameters=model.parameters(),
                             config = ds_config,
                             optimizer=optimizer)
        model.load_checkpoint(model_filename)
        print (f'load model: {model_filename} successfully!')
        print_model_params_num(model)
    # model.to(device)
    model.eval()
    input_token_ids = encode(input_text)
    input_texts = decode(input_token_ids).replace(' ', '*')
    print (f'input text is {input_text}, input of model is {input_texts}')
    generated_idx = model.generate(idx=torch.tensor(input_token_ids, dtype=torch.long, device=device).view(1, -1), max_new_tokens=max_new_tokens, decode=None)
    generated_text = decode(generated_idx[0].tolist())
    print ()
    print (generated_text)
    if save_output:
        save_text(generated_text, 'results/infer.txt')  

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
        save_text(generated_text, 'results/infer.txt')    
