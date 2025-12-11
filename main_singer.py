from model import BigramLanguageModel
from train_eval_utils import train_model, save_model, inference
from utils import print_model_params_num 

Train_and_Infer = True 

if __name__ == '__main__':
    if Train_and_Infer:
        model = BigramLanguageModel()
        print_model_params_num(model)
        model = train_model(model)
        model_filename =save_model(model, prefix='./ckpts')
        inference(model=model, model_filename=model_filename, input_text='请给我写一首诗', max_new_tokens=1000)
    else:
        model_filename = './ckpts/model_merged_file_20240525_095927.pth'
        print ('start inference ... ')
        inference(Model=BigramLanguageModel, model_filename=model_filename, input_text='please write a pome for me', max_new_tokens=1000)
