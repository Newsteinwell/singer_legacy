

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def human_readable_count(number):
    units = ['K', 'M', 'B', 'T']
    unit = ''
    for i in range(len(units)):
        if number >= 1000:
            number /= 1000
            unit = units[i]
        else:
            break
    return f"{number:.2f}{unit}" if unit else f"{number}"

def print_model_params_num(model):
    num_params = count_parameters(model)
    print (f'number of parameters is : {human_readable_count(num_params)}')
    