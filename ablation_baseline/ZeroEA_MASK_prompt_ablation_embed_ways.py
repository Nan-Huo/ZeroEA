import numpy as np 
import os
import torch
import scipy
import json
import string
from zhon.hanzi import punctuation
from scipy import spatial
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from levenshtein_distance import Levenshtein
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer
import banana_dev as banana


def knowledge_injection(ent_text, tokenizer, model):
    api_key="5463021c-fd79-4c8c-bf92-79ac68d42a41"
    model_key="gptj"
    text_input_gptj = "Define " + ent_text + " with Wikipedia."
    model_inputs = { "text": text_input_gptj, "length": 200, "temperature": 1, "topK": 1, "topP": 1}
    out = banana.run(api_key, model_key, model_inputs)
    output = out["modelOutputs"][0]["output"].replace("\n","")
    cut_pos = output.rfind(". ")
    mask_txt = "[MASK] is identical with " + ent_text +'. '
    if cut_pos < 10:
        injection_txt = ent_text + ": " + mask_txt + output
    else:
        injection_txt = ent_text + ": " + mask_txt + output[0:cut_pos+2]

    tokens = tokenizer(
        injection_txt,                  # 分词文本
        return_token_type_ids=False,    # 返回是前一句还是后一句
        return_attention_mask=True,     # 返回attention_mask
        return_tensors='pt')            # 返回pytorch类型的tensor

    tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    with torch.no_grad():    #将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
        encoded_layers = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])['last_hidden_state']
        last_hidden_state = encoded_layers[0]
#         ent_embed = last_hidden_state[0,:].numpy()
        ent_embed = get_embed(last_hidden_state, tokenized_text).numpy()
        
    return np.array(ent_embed)


def get_hits(Lvec, Rvec, entity_text_left, entity_text_right, entity_embed_left, entity_embed_right, embed_flg, top_k=(1, 10, 50, 100)):
#     sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='euclidean')
    
    # Enhance the entity text if very similar
    for i in range(Lvec.shape[0]):
        ent_cosine_dist = scipy.spatial.distance.cosine(entity_embed_left[i], entity_embed_right[i])
#         if ent_cosine_dist <= 0.1:
#             Lvec[i] = Lvec[i] + entity_embed_left[i]
#             Rvec[i] = Rvec[i] + entity_embed_right[i]
        if ent_cosine_dist <= 0.1:
            Lvec[i] = Lvec[i] + 2*entity_embed_left[i]
            Rvec[i] = Rvec[i] + 2*entity_embed_right[i]
        if 0.1 < ent_cosine_dist <= 0.2:
            Lvec[i] = Lvec[i] + entity_embed_left[i]
            Rvec[i] = Rvec[i] + entity_embed_right[i]
            
    sim = 1 - np.array(cosine_similarity(Lvec, Rvec))
    euc_dist = np.array(euclidean_distances(Lvec, Rvec))
    top_lr = [0] * len(top_k)
    ###: initialize failure case lists
    lr_fail_list = []
    rl_fail_list = []
    lr_fail_case = []
    rl_fail_case = []
    lr_fail_case_dict = {}
    rl_fail_case_dict = {}
    
#     print("++++++++++++++++++++++++++++++++ LR ++++++++++++++++++++++++++++++")
    RR_left = 0
    for i in range(Lvec.shape[0]):
        ### handle special cases:
        score_cnt = 0
        align_flg = False
        ent_l_list = entity_text_left[i].split(' ')
        ent_r_list = entity_text_right[i].split(' ')
        for item in ent_l_list:
            if item in ent_r_list:
                score_cnt += 1
                
        err_tole_flg = True
        score_cnt_2 = 0
        if len(ent_l_list) == len(ent_r_list):
            for j in range(len(ent_l_list)):
                if ent_l_list[j] == ent_r_list[j]:
                    score_cnt_2 += 1
                    continue
                    
                if (ent_l_list[j] in ent_r_list[j]) or (ent_r_list[j] in ent_l_list[j]):
                    if err_tole_flg:
                        score_cnt_2 += 1
                        err_tole_flg = False
           
        
        if (score_cnt == min(len(ent_l_list), len(ent_r_list))) or (score_cnt_2 == min(len(ent_l_list), len(ent_r_list))) or (entity_text_left[i].replace(' ','') in entity_text_right[i].replace(' ','')) or (entity_text_right[i].replace(' ','') in entity_text_left[i].replace(' ','')):
            align_flg = True
            
            
        rank = sim[i, :].argsort()
        rank_euc = euc_dist[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        rank_euc_index = np.where(rank_euc == i)[0][0]
        RR_left += 1/(rank_index+1)
        for j in range(len(top_k)):
            if rank_index < top_k[j] or rank_euc_index < top_k[j] or align_flg:  # We assume same entity txt to be aligned
#             if rank_index < top_k[j] or entity_text_left[i] == entity_text_right[i] or rank_euc_index < top_k[j]:  # We assume same entity txt to be aligned
                top_lr[j] += 1
                
            ###: add failure cases
            if (j == 0 and sim[i, rank[0]] > 0.5) or (j == 0 and rank_index >= top_k[j] and (not align_flg) and rank_euc_index >= top_k[j]):
#             if (j == 0 and sim[i, rank[0]] > 0.5) or (j == 0 and rank_index >= top_k[j] and entity_text_left[i] != entity_text_right[i] and  rank_euc_index >= top_k[j]):
                lr_fail_list.append(i)
                err_cosine_dist = scipy.spatial.distance.cosine(entity_embed_left[i], entity_embed_right[rank[0]])
                lr_fail_case.append((i, rank[0], sim[i, rank[0]], err_cosine_dist, entity_text_right[rank[0]], rank_index, sim[i, rank[rank_index]], entity_text_left[i], entity_text_right[i]))
                lr_fail_case_dict[i] = {"most_similar_idx": str(rank[0]), "most_similar_distance": str(sim[i, rank[0]]), "err_cosine_distance": str(err_cosine_dist), "mis-align_entity": entity_text_right[rank[0]], "rignt_ent_rank": str(rank_index), "rignt_ent_distance": str(sim[i, rank[rank_index]]), "left_entity": entity_text_left[i], "right_entity": entity_text_right[i]}
                
#                 print(str(i) + '\t' + entity_text_left[i] + '\t' + entity_text_right[i], flush=True)
#     print("-------------------------------- End LR ---------------------------")
    
#     print()
#     print()
    
#     print("++++++++++++++++++++++++++++++++ RL ++++++++++++++++++++++++++++++")
    top_rl = [0] * len(top_k)
    RR_right = 0
    for i in range(Rvec.shape[0]):
        ### handle special cases:
        score_cnt = 0
        align_flg = False
        ent_l_list = entity_text_left[i].split(' ')
        ent_r_list = entity_text_right[i].split(' ')
        for item in ent_l_list:
            if item in ent_r_list:
                score_cnt += 1
                
        err_tole_flg = True
        score_cnt_2 = 0
        if len(ent_l_list) == len(ent_r_list):
            for j in range(len(ent_l_list)):
                if ent_l_list[j] == ent_r_list[j]:
                    score_cnt_2 += 1
                    continue
                    
                if (ent_l_list[j] in ent_r_list[j]) or (ent_r_list[j] in ent_l_list[j]):
                    if err_tole_flg:
                        score_cnt_2 += 1
                        err_tole_flg = False
                

        if (score_cnt == min(len(ent_l_list), len(ent_r_list))) or (score_cnt_2 == min(len(ent_l_list), len(ent_r_list))) or (entity_text_left[i].replace(' ','') in entity_text_right[i].replace(' ','')) or (entity_text_right[i].replace(' ','') in entity_text_left[i].replace(' ','')):
            align_flg = True    
            
        rank = sim[:, i].argsort()
        rank_euc = euc_dist[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        rank_euc_index = np.where(rank_euc == i)[0][0]
        RR_right += 1/(rank_index+1)
        for j in range(len(top_k)):
            if rank_index < top_k[j] or align_flg or rank_euc_index < top_k[j]:
#             if rank_index < top_k[j] or entity_text_left[i] == entity_text_right[i] or rank_euc_index < top_k[j]:
                top_rl[j] += 1
                
            ### add
            if (j == 0 and sim[rank[0], i] > 0.5) or (j == 0 and rank_index >= top_k[j] and (not align_flg) and  rank_euc_index >= top_k[j]):
#             if (j == 0 and sim[rank[0], i] > 0.5) or (j == 0 and rank_index >= top_k[j] and entity_text_left[i] != entity_text_right[i] and  rank_euc_index >= top_k[j]):
                rl_fail_list.append(i)
                err_cosine_dist = scipy.spatial.distance.cosine(entity_embed_left[rank[0]], entity_embed_right[i])
                rl_fail_case.append((i, rank[0], sim[rank[0], i], err_cosine_dist, entity_text_left[rank[0]], rank_index, sim[rank[rank_index], i], entity_text_left[i], entity_text_right[i]))
                rl_fail_case_dict[i] = {"most_similar_idx": str(rank[0]), "most_similar_distance": str(sim[rank[0], i]), "err_cosine_distance": str(err_cosine_dist), "mis-align_entity": entity_text_left[rank[0]], "rignt_ent_rank": str(rank_index), "rignt_ent_distance": str(sim[rank[rank_index], i]), "left_entity": entity_text_left[i], "right_entity": entity_text_right[i]}
                
#                 print(str(i) + '\t' + entity_text_left[i] + '\t' + entity_text_right[i], flush=True)
#     print("-------------------------------- End RL ---------------------------")
    
    if embed_flg == 0:
        embed_way = "Mean of Only entity and first sentence."
    elif embed_flg == 1:
        embed_way = "Mean of All sentence."
    elif embed_flg == 2:
        embed_way = "Mean of The entity in all sentences."
    
    print("Embed way is: " + embed_way)
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_left / Lvec.shape[0] * 100))
    print()
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_right / Lvec.shape[0] * 100))
    
    print()
    print()    

    failure_list = list(set(lr_fail_list+rl_fail_list))
    
    return failure_list        


def get_embed_1(summed_last_4_layers, tokenized_text):
    sep_idx = tokenized_text.index(".")
    cls_idx = tokenized_text.index("[CLS]")
    summed_embed = summed_last_4_layers[cls_idx]
    for i in range(cls_idx+1, sep_idx): 
        summed_embed = summed_embed + summed_last_4_layers[i]
        
    return summed_embed

#     return summed_embed / len(range(cls_idx, sep_idx))


def get_embed_2(summed_last_4_layers, tokenized_text):
    cls_idx = tokenized_text.index("[CLS]")
    summed_embed = summed_last_4_layers[cls_idx]
    for i in range(len(summed_last_4_layers)): 
        summed_embed = summed_embed + summed_last_4_layers[i]
        
    return summed_embed

#     return summed_embed / (len(summed_last_4_layers)+1)


def get_embed_3(summed_last_4_layers, tokenized_text):
    sep_idx = tokenized_text.index(":")
    cls_idx = tokenized_text.index("[CLS]")
    ent_tokens = tokenized_text[cls_idx+1 : sep_idx]
    idx_list = [i for i,x in enumerate(tokenized_text) if x in ent_tokens ]
    summed_embed = summed_last_4_layers[cls_idx]
    for j in idx_list: 
        summed_embed = summed_embed + summed_last_4_layers[j]

    return summed_embed


def get_target_embed(filename, tokenizer, model, embed_flg):
    with open(filename, "r") as input_f:
        entity_text_all = []
        target_embed = []
        entity_embed_all = []
        counter = 0
        counter_list = []
        for line in input_f:
            tmp = line.strip().replace("...", "")
            real_label = tmp.replace("\n", '')
            
            ### Adding wrod_embed
            entity_text = real_label.split(":")[0].strip()
            entity_text = entity_text.split('(')[0].strip()
            entity_text = entity_text.split('（')[0].strip()
            
            # remove punctuation
            punctuation_eng = string.punctuation
            punctuation_zh = punctuation
            for i in punctuation_eng:
                entity_text = entity_text.replace(i, '')
                
            for j in punctuation_zh:
                entity_text = entity_text.replace(j, '')
                
                
            sep_idx = real_label.index(":")
            real_label = entity_text + real_label[sep_idx:]
            input_txt_list = real_label.split(":")
            input_txt_all = entity_text + ": " + "[MASK] is identical with " + entity_text +'. '
            for i in input_txt_list[1:]:
                input_txt_all = input_txt_all + i
                
            entity_text_all.append(entity_text)
            
            input_txt_ent = "[MASK] is identical with " + entity_text +'. '
            tokens_ent = tokenizer(
                input_txt_ent,                  # 分词文本
                return_token_type_ids=False,    # 返回是前一句还是后一句
                return_attention_mask=True,     # 返回attention_mask
                return_tensors='pt')            # 返回pytorch类型的tensor
            
            tokenized_ent_text = tokenizer.convert_ids_to_tokens(tokens_ent["input_ids"][0])
            encoded_layers_ent = model(input_ids=tokens_ent["input_ids"], attention_mask=tokens_ent["attention_mask"])['last_hidden_state']
            last_hidden_state_ent = encoded_layers_ent[0]
            mask_idx = tokenized_ent_text.index("[MASK]")
            entity_embed = last_hidden_state_ent[mask_idx]
            entity_embed_all.append(entity_embed.detach().numpy())
            
            tokens = tokenizer(
                input_txt_all,                  # 分词文本
                return_token_type_ids=False,    # 返回是前一句还是后一句
                return_attention_mask=True,     # 返回attention_mask
                return_tensors='pt')            # 返回pytorch类型的tensor
            
            tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

            with torch.no_grad():    #将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
                try:
                    encoded_layers = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])['last_hidden_state']
                    last_hidden_state = encoded_layers[0]
                    
                    ### different embed ways
                    if embed_flg == 0:
                        tokens = get_embed_1(last_hidden_state, tokenized_text)
                    elif embed_flg == 1:
                        tokens = get_embed_2(last_hidden_state, tokenized_text)
                    elif embed_flg == 2:
                        tokens = get_embed_3(last_hidden_state, tokenized_text)
                        
                    target_embed.append(tokens.numpy())
                except:
                    counter_list.append(counter)
                    target_embed.append(len(entity_embed)*[0])
            
            if counter > thresh_num:
                break
            counter += 1
            
    return target_embed, counter_list, entity_text_all, entity_embed_all
    
thresh_num = 15000
bert_model = 'bert-base-uncased'  #"bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
model.eval()  #这里是验证模型，可以节省很多不必要的反向传播
for embed_flg in range(3):
    target_embed_1, c_list_1, entity_text_left, entity_embed_left = get_target_embed('text_input_no_train_11_wxt_KI.txt', tokenizer, model, embed_flg)
    target_embed_2, c_list_2, entity_text_right, entity_embed_right = get_target_embed('text_input_no_train_22_wxt_KI.txt', tokenizer, model, embed_flg)


    ### Del too long cases
    tmp_list = list(set(c_list_1 + c_list_2))
    c_list_all = sorted(tmp_list, reverse=True)
    print("--------------------------------------------------")
    print(c_list_all)
    if c_list_all:
        for j in c_list_all:
            del target_embed_1[j]
            del target_embed_2[j]
            del entity_text_left[j]
            del entity_text_right[j]
            del entity_embed_left[j]
            del entity_embed_right[j]

    print("=====================================================")
    Lvec = np.array(target_embed_1)
    Rvec = np.array(target_embed_2)

    failure_list = get_hits(Lvec, Rvec, entity_text_left, entity_text_right, entity_embed_left, entity_embed_right, embed_flg)







