from load_files import *

def SetData(input_name, output_name, domain_name, emo_name,dev_num,train_num):
    
    input_lang, output_lang, pair = prepareData(input_name,output_name,domain_name)
    #test_numはtestとdevの残り
    train_pairs = pair[:train_num]
    dev_pairs = pair[train_num:train_num + dev_num]
    test_pairs = pair[train_num + dev_num :]

    emo_tensor = LoadEmo(domain_name,emo_name)
    return input_lang, output_lang, train_pairs, dev_pairs, test_pairs, emo_tensor

