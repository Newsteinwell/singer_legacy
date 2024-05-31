from model import BigramLanguageModel
from train_eval_utils import train_model, train_model_ds, save_model_ds, inference, inference_ds
from utils import print_model_params_num 
import time
import torch
import os

Train_and_Infer = True 
# Specify which GPUs to use
#os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'  # Use GPUs

# command line, run: deepspeed --include localhost:3,4,5,6 main_singer_DP.py 
if __name__ == '__main__':
    if Train_and_Infer:
        model = BigramLanguageModel()
        print_model_params_num(model)
        start_time = time.time()
        model = train_model_ds(model)
        rank = torch.distributed.get_rank()
        print (f'rank: {rank}, training time is {time.time() - start_time:.2f} s')
        model_filename =save_model_ds(model, prefix='./ckpts_ds', suffix='entire_epoch')
        inference_ds(model=model, model_filename=model_filename, input_text='请给我写一首诗', max_new_tokens=1000)

    else:
        model_filename = './ckpts_ds/model_songs_poetry_merged_file_20240527_191654.pth'
        print ('start inference ... ')
        start_time = time.time()
        inference_ds(Model=BigramLanguageModel, model_filename=model_filename, input_text='单身情歌', max_new_tokens=1000)
        rank = torch.distributed.get_rank()
        print (f'rank: {rank}, inference time is {time.time() - start_time:.2f} s')
