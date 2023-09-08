import numpy as np 
import os
import sys
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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge 


def web_search(ent_text, tokenizer, model):
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


def get_hits(Lvec, Rvec, entity_text_left, entity_text_right, entity_embed_left, entity_embed_right, rouge_thresh, top_k=(1, 10, 50, 100)):
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
    rouge_score_L = []
    rouge_score_R = []
    rouge_score_L_all = []
    rouge_score_R_all = []
    
    # print("++++++++++++++++++++++++++++++++ LR ++++++++++++++++++++++++++++++")
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
                
        if (score_cnt == min(len(ent_l_list), len(ent_r_list))) or (entity_text_left[i] in entity_text_right[i]) or (entity_text_right[i] in entity_text_left[i]):
            align_flg = True
        
        ### KI
        rouger = Rouge()
        try:
            rouge_score = rouger.get_scores(entity_text_left[i], entity_text_right[i])[0]["rouge-l"]["f"]
            rouge_score_L_all.append(rouge_score)
            if rouge_score > rouge_thresh:
                align_flg = True
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(entity_text_left[i] + '\t' + entity_text_right[i] + '\t' + str(i))
            print()
            
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
                
                rouge_score_L.append(rouge_score)
                # print(str(i) + '\t' + entity_text_left[i] + '\t' + entity_text_right[i] + '\t' + '\t' + str(rouge_score), flush=True)
    # print("-------------------------------- End LR ---------------------------")
    
    # print()
    # print()
    
    # print("++++++++++++++++++++++++++++++++ RL ++++++++++++++++++++++++++++++")
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
                
        if (score_cnt == min(len(ent_l_list), len(ent_r_list))) or (entity_text_left[i] in entity_text_right[i]) or (entity_text_right[i] in entity_text_left[i]):
            align_flg = True    
            
        ### KI
        rouger = Rouge()
        try:
            rouge_score = rouger.get_scores(entity_text_left[i], entity_text_right[i])[0]["rouge-l"]["f"]
            rouge_score_R_all.append(rouge_score)
            if rouge_score > rouge_thresh:
                align_flg = True
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(entity_text_left[i] + '\t' + entity_text_right[i] + '\t' + str(i))
            print()
            
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
                
                rouge_score_R.append(rouge_score)
                # print(str(i) + '\t' + entity_text_left[i] + '\t' + entity_text_right[i] + '\t' + '\t' + str(rouge_score), flush=True)
    # print("-------------------------------- End RL ---------------------------")
                
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_left / Lvec.shape[0] * 100))
    print()
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_right / Lvec.shape[0] * 100))

    ##: print Rouge score
    print("xxxxxxxxxxxxxxxxxxxxxxx: Rouge Score xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("For Left")
    print("avg_all:", np.mean(np.array(rouge_score_L_all)))
    print("avg_wrong:", np.mean(np.array(rouge_score_L)))
    print("median_all:", np.median(np.array(rouge_score_L_all)))
    print("median_wrong:", np.median(np.array(rouge_score_L)))
    print("min_all:", np.min(np.array(rouge_score_L_all)))
    print("max_wrong:", np.max(np.array(rouge_score_L)))
    print()
    print("For right")
    print("avg_all:", np.mean(np.array(rouge_score_R_all)))
    print("avg_wrong:", np.mean(np.array(rouge_score_R)))
    print("median_all:", np.median(np.array(rouge_score_R_all)))
    print("median_wrong:", np.median(np.array(rouge_score_R)))
    print("min_all:", np.min(np.array(rouge_score_R_all)))
    print("max_wrong:", np.max(np.array(rouge_score_R)))
    
    
    ### If low confidence, then knowledge injection.
    failure_list = list(set(lr_fail_list+rl_fail_list))
    
    return failure_list        


def get_embed(summed_last_4_layers, tokenized_text):
#     mask_idx = tokenized_text.index("[MASK]")
#     cls_idx = tokenized_text.index("[CLS]")
#     summed_embed = summed_last_4_layers[mask_idx] + summed_last_4_layers[cls_idx]
    sep_idx = tokenized_text.index(".")
    cls_idx = tokenized_text.index("[CLS]")
    summed_embed = summed_last_4_layers[cls_idx]
    for i in range(cls_idx+1, sep_idx): 
        summed_embed = summed_embed + summed_last_4_layers[i]

    return summed_embed


def KI_checker(fn_1, fn_2, threshold):
    context_1 = {}
    context_2 = {}
    KI_idx_list = []
    with open(fn_1, "r") as input_1:
        counter_1 = 0
        for line in input_1:
            tmp = line.strip()
            real_label_KI = tmp.replace("\n", '')
            entity_text = real_label_KI.split(":")[0].strip()
            # remove punctuation
            punctuation_eng = string.punctuation
            punctuation_zh = punctuation
            for i in punctuation_eng:
                entity_text = entity_text.replace(i, '')

            for j in punctuation_zh:
                entity_text = entity_text.replace(j, '')
                
            context_1[counter_1] = entity_text
            counter_1 += 1
        
    with open(fn_2, "r") as input_2:
        counter = 0
        for line in input_2:
            tmp = line.strip()
            real_label_KI = tmp.replace("\n", '')
            entity_text = real_label_KI.split(":")[0].strip()
            # remove punctuation
            punctuation_eng = string.punctuation
            punctuation_zh = punctuation
            for i in punctuation_eng:
                entity_text = entity_text.replace(i, '')

            for j in punctuation_zh:
                entity_text = entity_text.replace(j, '')
                
            context_2[counter] = entity_text
            counter += 1

    rouger = Rouge()
    for i in range(counter_1):
        scores = rouger.get_scores(context_1[i], context_2[i])
        rouge_l = scores[0]["rouge-l"]["f"]
        if rouge_l < threshold:
            KI_idx_list.append(i)
        
    KI_ratio = (len(KI_idx_list) / counter_1) * 100
    
    return KI_idx_list, KI_ratio
            
            
def get_target_embed(filename, fn_ref, KI_idx_list, tokenizer, model, KI_flg):
    ### KI
    if KI_flg:
        ref_context_KI = {}
        with open(fn_ref, "r") as input_ref:
            counter_KI = 0
            for line in input_ref:
                tmp = line.strip()
                real_label_KI = tmp.replace("\n", '')
                ref_context_KI[counter_KI] = real_label_KI 
                if len(real_label_KI) < 1:
                    # print("!!!!!!!!!!!!! In loading process: !!!!!!!!!!!!!!!!!!!!!")
                    # print(tmp)
                    # print(counter_KI)
                    raise AssertionError
                counter_KI += 1
    
    
    with open(filename, "r") as input_f:
        entity_text_all = []
        target_embed = []
        entity_embed_all = []
        counter = 0
        counter_list = []
        for line in input_f:
            tmp = line.strip().replace("...", "")
            real_label = tmp.replace("\n", '')
            
            ### KI
            if KI_flg:
                if counter in KI_idx_list:
                    real_label = ref_context_KI[counter]
                    if len(real_label) < 1:
                        # print("!!!!!!!!!!!!! In replace process: !!!!!!!!!!!!!!!!!!!!!")
                        # print(tmp)
                        # print(counter)
                        raise AssertionError

            ### Adding wrod_embed
            entity_text_origin = real_label.split(":")[0].strip()
            entity_text = entity_text_origin.split('(')[0].strip()
            entity_text = entity_text.split('（')[0].strip()
            
            if len(entity_text) < 1:
                entity_text = entity_text_origin.split(')')[-1].strip()
                if len(entity_text) < 1:
                    # print("!!!!!!!!!!!!! In entity extraction process: !!!!!!!!!!!!!!!!!!!!!")
                    # print(real_label)
                    # print(counter)
                    # print()
                    # print()

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
#                     tokens = last_hidden_state[0,:]
                    tokens = get_embed(last_hidden_state, tokenized_text)
                    target_embed.append(tokens.numpy())
                except:
                    counter_list.append(counter)
                    target_embed.append(len(entity_embed)*[0])

            if counter > thresh_num:
                break
            counter += 1
            
    return target_embed, counter_list, entity_text_all, entity_embed_all
    
thresh_num = 15000
threshold_KI = int(sys.argv[1]) / 100
input_prompt_dir_KI_1 = sys.argv[2]   
input_prompt_dir_KI_2 = sys.argv[3] 
input_prompt_dir_1 = sys.argv[4]   
input_prompt_dir_2 = sys.argv[5] 
print("threshold_KI is: ", threshold_KI, flush=True)
bert_model = 'bert-base-uncased'  #"bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
model.eval()  #这里是验证模型，可以节省很多不必要的反向传播

KI_idx_list, KI_ratio = KI_checker(input_prompt_dir_1, input_prompt_dir_2, threshold_KI)
target_embed_1, c_list_1, entity_text_left, entity_embed_left = get_target_embed(input_prompt_dir_1, input_prompt_dir_KI_1, KI_idx_list, tokenizer, model, True)
target_embed_2, c_list_2, entity_text_right, entity_embed_right = get_target_embed(input_prompt_dir_2, input_prompt_dir_KI_2, KI_idx_list, tokenizer, model, False)


### Del too long cases
tmp_list = list(set(c_list_1 + c_list_2))
c_list_all = sorted(tmp_list, reverse=True)
# print("--------------------------------------------------")
# print(c_list_all)
if c_list_all:
    for j in c_list_all:
        del target_embed_1[j]
        del target_embed_2[j]
        del entity_text_left[j]
        del entity_text_right[j]
        del entity_embed_left[j]
        del entity_embed_right[j]

# print("=====================================================")
Lvec = np.array(target_embed_1)
Rvec = np.array(target_embed_2)
# First try
if threshold_KI >= 0.7:
    threshold_rouge = threshold_KI
else:
    threshold_rouge = 0.8
failure_list = get_hits(Lvec, Rvec, entity_text_left, entity_text_right, entity_embed_left, entity_embed_right, threshold_rouge)


print("******************************************** KI Ratio: % *********************************************************")
print(KI_ratio)

# Knowledge injection
for i in failure_list:
    ent_text_l = entity_text_left[i]
    ent_text_r = entity_text_right[i]
    ent_embed_l = web_search(ent_text_l, tokenizer, model)
    ent_embed_r = web_search(ent_text_r, tokenizer, model)
    injection_cosine_dist = scipy.spatial.distance.cosine(ent_embed_l, ent_embed_r)
    if injection_cosine_dist < 0.1:
        Lvec[i] = Lvec[i] + 2*ent_embed_l
        Rvec[i] = Rvec[i] + 2*ent_embed_r
    else:
        Lvec[i] = Lvec[i] + ent_embed_l
        Rvec[i] = Rvec[i] + ent_embed_r

failure_list_new = get_hits(Lvec, Rvec, entity_text_left, entity_text_right, entity_embed_left, entity_embed_right)






