import datetime
import shutil
import argparse
import os
import logging
import random
import datetime

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from tqdm import tqdm, trange

from transformers import BertJapaneseTokenizer as BertTokenizer

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import CONFIG_NAME, WEIGHTS_NAME
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data import make_dataloader
from LMmodel import BertMouth
from bert_mouth import save, initialization_text


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available.")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="A random seed for initialization.")
    parser.add_argument('--seq_length',
                        type=int,
                        default=30,
                        help="The sequence length generated.")
    parser.add_argument('--loop', type=int, default=5)
    parser.add_argument('--inputmsg', type=str,
                        default="今日は暑いですね")
    parser.add_argument("--fix_word", default="暑", type=str,
                        help="A fixed word in text generation.")
    parser.add_argument('--modes', type=str,
                        default="forward,quiet")

    args = parser.parse_args()
    return args

def mydecode(generated_token_ids, input_dict):
  length = input_dict['max_length']
  tokenizer = input_dict['tokenizer']

  sampled_sequence = [tokenizer.ids_to_tokens[token_id]
                        for token_id in generated_token_ids[0].cpu().numpy()]
  sampled_sequence = "".join([token[2:] if token.startswith("##") else token
                                for token in sampled_sequence[0:length + 1]])
  
  return sampled_sequence


def my_tokenizer(input_dict):
  tokenizer = input_dict['tokenizer']
  words = input_dict['fix_word']

  init_tokens = []
  init_tokens.append(tokenizer.vocab["[CLS]"])
  
  for word in words:
    t1 = tokenizer.tokenize(word)
    t2 = tokenizer.convert_tokens_to_ids(t1)
    
    init_tokens.extend(t2)
  init_tokens.append(tokenizer.vocab["[SEP]"])

  t3 = tokenizer.convert_ids_to_tokens(init_tokens)
  return init_tokens

# 固定ワードの埋め込み
def embedding(input_dict):
  tokenizer = input_dict['tokenizer']
  fix_words = input_dict['fix_word']
  generated_token_ids = input_dict['generated_token_ids']
  length = input_dict['length']
  max_length = input_dict['max_length']

  first_len = input_dict['first_len']
  length = length + first_len

  fix_word_interval = set()
  for fix_word in fix_words:
    tokenized_fix_word = tokenizer.tokenize(fix_word)
    
    fix_word_pos = random.randint(
      first_len+1,
      length - len(tokenized_fix_word))
    temp_set = set(
      range(fix_word_pos,
      fix_word_pos + len(tokenized_fix_word)))
    while not temp_set.isdisjoint(fix_word_interval):
      fix_word_pos = random.randint(
        first_len+1,
        length - len(tokenized_fix_word))
      temp_set = set(
        range(fix_word_pos,
        fix_word_pos + len(tokenized_fix_word)))

    fix_word_interval = fix_word_interval.union(temp_set)
    
    for i in range(len(tokenized_fix_word)):
      generated_token_ids[fix_word_pos + i] = \
        tokenizer.convert_tokens_to_ids(tokenized_fix_word[i])

  input_dict['tokenized_fix_word'] = tokenized_fix_word
  input_dict['fix_word_interval'] = set(fix_word_interval)
  input_dict['generated_token_ids'] = generated_token_ids

  return input_dict

def create_mask(input_dict):
  device = input_dict['device']
  generated_token_ids = input_dict['generated_token_ids']
  max_length = input_dict['max_length']

  input_type_id = [0] * max_length
  input_mask = [1] * len(generated_token_ids)
  while len(input_mask) < max_length:
    generated_token_ids.append(0)
    input_mask.append(0)

  generated_token_ids = torch.tensor([generated_token_ids],
                                    dtype=torch.long).to(device)
  input_type_id = torch.tensor(
    [input_type_id], dtype=torch.long).to(device)
  input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
  
  input_dict['generated_token_ids'] = generated_token_ids
  input_dict['input_type_id'] = input_type_id
  input_dict['input_mask'] = input_mask
  
  return input_dict

def generate_core(j,input_dict, quiet=False):
  tokenizer = input_dict['tokenizer']
  model = input_dict['model']
  fix_word_interval = input_dict['fix_word_interval']
  generated_token_ids = input_dict['generated_token_ids']
  input_type_id = input_dict['input_type_id']
  input_mask = input_dict['input_mask']

  if fix_word_interval:
    if j + 1 in fix_word_interval:
      return input_dict
  
  generated_token_ids[0, j + 1] = tokenizer.vocab["[MASK]"]
  if quiet:
    print(str(j) + '-msk: ' + mydecode(generated_token_ids,input_dict))
  
  outputs = model(input_ids=generated_token_ids, 
                  token_type_ids=input_type_id,
                  attention_mask=input_mask, 
                  masked_lm_labels=generated_token_ids, 
                  lm_labels=generated_token_ids)
  logits = outputs[0][0]
  
  sampled_token_id = torch.argmax(logits[j + 1])
  sampled_token_ids = []
  for r in logits:
    sampled_token_ids.append(torch.argmax(r))
  
  generated_token_ids[0, j + 1] = sampled_token_id
  
  if quiet:
    print(str(j) + '-out: ' + mydecode(torch.tensor([sampled_token_ids]),input_dict))
    print(str(j) + '-fix: ' + mydecode(generated_token_ids, input_dict))

  input_dict['generated_token_ids'] = generated_token_ids
  
  return input_dict

def generate(input_dict, quiet=False):
  length = input_dict['max_length'] -1
  first_len = input_dict['first_len']
  for j in range(first_len-1, length):
    input_dict = generate_core(j, input_dict, quiet)    
  return input_dict

# reverse
def generate_rev(input_dict, quiet=False):
  length = input_dict['max_length']-1
  first_len = input_dict['first_len']
  for j in reversed(range(first_len-1, length)):
    input_dict = generate_core(j, input_dict, quiet)    
  return input_dict

# random
def generate_rand(input_dict, quiet=False):
  length = input_dict['max_length']-1
  first_len = input_dict['first_len']
  my_index = [i for i in range(first_len-1,length)]
  
  random.shuffle(my_index)
  for j in my_index:
    input_dict = generate_core(j, input_dict, quiet)    
  return input_dict

def create_input(input_dict):
  input_dict = embedding(input_dict)
  input_dict = create_mask(input_dict)
  return input_dict

def init_save(LOG_DIR):
  now = datetime.datetime.now()
  savefile = now.strftime('%Y-%m-%d') + '.log'

  if os.path.exists(os.path.join(LOG_DIR,savefile)):
    shutil.copy(os.path.join(LOG_DIR,savefile), savefile)
  elif not os.path.exists(savefile):
    os.makedirs(LOG_DIR)
  with open(savefile, 'w', encoding='utf-8') as f:
    f.write(savefile)
    f.write('\n----\n')

  return os.path.abspath(savefile)

def save_conv(savefile, msg):
  with open(savefile, 'a', encoding='utf-8') as f:
    if type(msg) == str:
      f.write(msg)
    else:
      f.write('\n'.join(msg))
    f.write('\n----\n')


def reply_core(input_dict, modes):
  temp = my_tokenizer(input_dict)
  first_len = len(temp)
  input_dict['first_len'] = first_len
  input_dict['length'] = input_dict['max_length'] - first_len - 2
  keywords = input_dict['keywords']
  keynum = input_dict['keynum']

  if type(input_dict['fix_word']) is str:
    input_dict['fix_word'] = [input_dict['fix_word']]
  elif type(input_dict['fix_word']) is not list:
    input_dict['fix_word'] = [""]

  temp.extend(initialization_text(
      input_dict["tokenizer"], 
      input_dict["length"],
      input_dict["fix_word"])[1:])

  input_dict['generated_token_ids'] = temp
  input_dict = create_input(input_dict)

  result = 'input: '
  result += mydecode(input_dict['generated_token_ids'], input_dict) + '\n'

  quiet = not("quiet" in modes)

  for i in range(input_dict["loop"]):
    if "forward" in modes:
      input_dict = generate(input_dict, quiet)
    if "reverse" in modes:
      input_dict = generate_rev(input_dict, quiet)
    if "random" in modes:
      input_dict = generate_rand(input_dict, quiet)
    if quiet:
      print('loop: ' + str(i))
      print(mydecode(input_dict['generated_token_ids'], input_dict))

  result_temp = mydecode(input_dict['generated_token_ids'], input_dict)
  result += 'raw output: ' + result_temp + '\n\n'

  if len(result_temp.split('[SEP]')) > 1:
    result_temp = result_temp.split('[SEP]')[1]
  result += 'output: ' + result_temp
  
  return result_temp, result


def load_models(model):
  device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = BertTokenizer.from_pretrained(
      model, do_lower_case=False,
      tokenize_chinese_chars=False)

  model_state_dict = torch.load(
    os.path.join(model, "pytorch_model.bin"),
    map_location=device)
  model = BertMouth.from_pretrained(
    model,
    state_dict=model_state_dict,
    num_labels=tokenizer.vocab_size)
  model.to(device)

  models = {
      "model": model,
      "tokenizer": tokenizer,
      "device": device
  }
  return models

def hello(input_dict, models, modes):
  inputmsg = input_dict["fix_word"][0]
  keyword = input_dict["keywords"]

  if '!' in inputmsg:
    inputmsg = inputmsg.replace('!', '！')
  elif '?' in inputmsg:
    inputmsg = inputmsg.replace('?', '？')

  if type(keyword) is not list:
    keyword = [keyword]
  
  input_dict["fix_word"] = [inputmsg]
  input_dict["keywords"] = keyword

  result, output = reply_core(input_dict, modes)

  LOG_DIR = "logs"
  save_conv(init_save(LOG_DIR), [inputmsg, output])
  if not("quiet" in modes):
    print(output)
  return result, output


if __name__ == "__main__":
    args = parse_argument()

    if args.seed is not -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    inputmsg = args.inputmsg
    keywords = [args.fix_word]
    modes = args.modes.split(",")

    input_dict = {
        'fix_word':[inputmsg],
        'length':5,
        'max_length':30,
        'loop':5,
        'model':models["model"],
        'tokenizer':models["tokenizer"],
        'device':models["device"],
        'keywords':keywords,
        'keynum': 1
    }
    

    if (not "models" in locals()) or (not "models" in globals()):
      models = load_models(args.bert_model)

    print(hello(input_dict, models, modes)[0])