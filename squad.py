# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:14:23 2020

@author: lenovo
"""

import sentencepiece as spm
import six
import unicodedata
import json
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from transformers import XLNetTokenizer
sp_model=spm.SentencePieceProcessor()
sp_model.Load("spiece.model")
random.seed(42)
max_seq_length = 550
max_query_length = 60
def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"')

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  # return_unicode is used only for py2

  # note(zhiliny): in some systems, sentencepiece only accepts str for py2
  if six.PY2 and isinstance(text, unicode):
    text = text.encode('utf-8')

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          piece[:-1].replace(SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode('utf-8')
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces
def encode_ids(sp_model, text, sample=False):
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  print(pieces)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids



class SquadFeatures(object):
    def __init__(
        self,
        
        input_ids,
        attention_mask,
        token_type_ids,
        start_position,
        end_position,
    ):
       
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position

def read_squad_examples(zhidao_input_file, search_input_file, is_training=True):
  
    examples = []

    with open(search_input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            if (len(source['answer_spans']) == 0):
                continue
            if source['answers'] == []:
                continue
            if (source['match_scores'][0] < 0.8):
                continue
            if (source['answer_spans'][0][1] > max_seq_length):
                continue
            if (source['fake_answers']==[]):
              print("empty_answers",source['question_id'])
              continue
            
            docs_index = source['answer_docs'][0]

            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1          ## !!!!!

            try:
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            doc = source['documents'][docs_index]['paragraphs'][answer_passage_idx]
            doc=preprocess_text(doc,lower=False)
            doc_tokens = encode_pieces(sp_model, doc, return_unicode=False)
            
            title=source['documents'][docs_index]['title']
            title=preprocess_text(title,lower=False)
            title_tokens=encode_pieces(sp_model, title, return_unicode=False)
            title_len = len(title_tokens) 
            #doc_tokens = doc_tokens[title_len:]
            if doc_tokens==[]:
              print(source['question_id'])
            #if doc_tokens[0]=='。':
            #  doc_tokens=doc_tokens[1:]
            #start_id , end_id = start_id -  title_len, end_id - title_len
            question=source['question']
            question=preprocess_text(question,lower=False)
            question_tokens=encode_pieces(sp_model, question, return_unicode=False)
            answer=source['fake_answers'][0]
            answer=preprocess_text(answer,lower=False)
            answer_tokens=encode_pieces(sp_model, answer, return_unicode=False) 
            if answer_tokens[0]=='▁':
              answer_tokens=answer_tokens[1:]
            if '▁' in answer_tokens[0]:
              answer_tokens[0]=answer_tokens[0][1:]
            flag=False  
            for start_idx in range(len(doc_tokens)):
                if doc_tokens[start_idx] == answer_tokens[0]:
                        point=start_idx
                        if len(doc_tokens[point:])<len(answer_tokens):
                            flag=False
                            break
                        if answer_tokens[len(answer_tokens)-1]=='。':
                            del answer_tokens[len(answer_tokens)-1]
                        for j in range(len(answer_tokens)):
                            if doc_tokens[point+j]==answer_tokens[j]:
                                flag=True
                                continue
                            else:
                                flag=False
                                break
                if flag==True:
                        break
                if flag==False:
                    continue
          
            if flag==True:
              start_id=start_idx  
              end_id=start_id+len(answer_tokens)-1
            if flag==False:
           
              continue                  
            if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
              continue

            if is_training:
          #      new_end_id = new_end_id - 1
                example = {
                        "qas_id":source['question_id'],
                        "question_text":question_tokens,
                     
                        "doc_tokens":doc_tokens,
                        "start_position":start_id,
                        "end_position":end_id}     

                examples.append(example)
        print(len(examples))
    with open(zhidao_input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):

            source = json.loads(line.strip())
            if (len(source['answer_spans']) == 0):
                continue
            if source['answers'] == []:
                continue
            if (source['match_scores'][0] < 0.8):
                continue
            if (source['answer_spans'][0][1] > max_seq_length):
                continue
            if (source['fake_answers']==[]):
              print("empty_answers",source['question_id'])
              continue
            if (source['question_id']==102820):
              continue
            docs_index = source['answer_docs'][0]

            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1          ## !!!!!


            try:
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            doc = source['documents'][docs_index]['paragraphs'][answer_passage_idx]
            doc=preprocess_text(doc,lower=False)
            doc_tokens = encode_pieces(sp_model, doc, return_unicode=False)
            
            title=source['documents'][docs_index]['title']
            title=preprocess_text(title,lower=False)
            title_tokens=encode_pieces(sp_model, title, return_unicode=False)
            title_len = len(title_tokens) 
            #doc_tokens = doc_tokens[title_len:]
            if doc_tokens==[]:
              print(source['question_id'])
            #if doc_tokens[0]=='。':
            #  doc_tokens=doc_tokens[1:]
            #start_id , end_id = start_id -  title_len, end_id - title_len
            question=source['question']
            question=preprocess_text(question,lower=False)
            question_tokens=encode_pieces(sp_model, question, return_unicode=False)
            answer=source['fake_answers'][0]
            answer=preprocess_text(answer,lower=False)
            answer_tokens=encode_pieces(sp_model, answer, return_unicode=False) 
            if answer_tokens[0]=='▁':
              answer_tokens=answer_tokens[1:]
            if '▁' in answer_tokens[0]:
              answer_tokens[0]=answer_tokens[0][1:]
            flag=False
            if doc_tokens==[]:
              print(source['question_id']) 
             
            for start_idx in range(len(doc_tokens)):
                if doc_tokens[start_idx] == answer_tokens[0]:
                        point=start_idx
                        if len(doc_tokens[point:])<len(answer_tokens):
                            flag=False
                            break
                        if answer_tokens[len(answer_tokens)-1]=='。':
                            del answer_tokens[len(answer_tokens)-1]
                        for j in range(len(answer_tokens)):
                            if doc_tokens[point+j]==answer_tokens[j]:
                                flag=True
                                continue
                            else:
                                flag=False
                                break
                if flag==True:
                        break
                if flag==False:
                    continue
            if flag==True:
              start_id=start_idx  
              end_id=start_id+len(answer_tokens)-1
            if flag==False:
              continue                  
            #if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
            #  continue
                
            if is_training:
          #      new_end_id = new_end_id - 1
                example = {
                        "qas_id":source['question_id'],
                        "question_text":question_tokens,
                     
                        "doc_tokens":doc_tokens,
                        "start_position":start_id,
                        "end_position":end_id}     

                examples.append(example)
      

    print("len(examples):",len(examples))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):

    features = []

    for example in tqdm(examples):
        query_tokens = list(example['question_text'])
        #question_type = example['question_type']    

        doc_tokens = example['doc_tokens']
        start_position = example['start_position']
        end_position = example['end_position']

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append("[SEP]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_length]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(segment_ids)
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask=[1]*len(input_ids)
            for i in range(max_seq_length-len(input_ids)):
              input_ids.append(0)
              segment_ids.append(1)
              attention_mask.append(0)
            assert len(input_ids)==len(segment_ids)


        features.append(SquadFeatures(
                        input_ids,
                        attention_mask,
                        token_type_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position, ))

        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
            )   
    cached_features_file="cached_train_xlnet_model.ckpt.index_550"
    torch.save({"dataset": dataset}, cached_features_file)

if __name__ == "__main__":

    tokenizer=XLNetTokenizer('spiece.model',do_lower_case=True)
    # 生成训练数据， train.data
    examples = read_squad_examples(zhidao_input_file='DuReader/data/extracted/trainset/zhidao.train.json',search_input_file='DuReader/data/extracted/trainset/search.train.json')
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                              max_seq_length=max_seq_length, max_query_length=max_query_length)
    # 生成验证数据， dev.data。记得注释掉生成训练数据的代码，并在196行将train.data改为dev.data
    # = read_squad_examples(zhidao_input_file='DuReader/data/extracted/devset/zhidao.dev.json',
    #                               search_input_file='DuReader/data/extracted/devset/search.dev.json')
    #features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
    #                                       max_seq_length=max_seq_length, max_query_length=max_query_length)
           
