#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import re
import sys
from transformers import BertTokenizer
from googletrans import Translator
import banana_dev as banana
import requests
from urllib import parse
import json
import networkx as nx
    
    
def text_save(filename, data):
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i])#.replace('[','').replace(']','')
        s = s + "\n"
        file.write(s)
    file.close()


def node_degree(target_node, node_dict, decay_factor_base, prev_neighbor_list, prev_neighbor_degree_list, KG_i, num_hop_i):
    decay_factor = decay_factor_base**(num_hop_i-1)
    all_neighbor_list = []
    neighbor_list_degree = []
    for i in range(len(node_dict[target_node])):
        neighbor_i = node_dict[target_node][i][0]
        if neighbor_i in prev_neighbor_list:
            idx_neighbor_i = prev_neighbor_list.index(neighbor_i)
            if neighbor_i != prev_neighbor_degree_list[idx_neighbor_i][0]:
                raise RuntimeError('neighbor_Error')
                
            degree_i = decay_factor * KG_i.degree[neighbor_i]
            if prev_neighbor_degree_list[idx_neighbor_i][-1] < degree_i:
                prev_neighbor_degree_list[idx_neighbor_i][-1] = degree_i
            
        else:
            if neighbor_i not in all_neighbor_list:
                all_neighbor_list.append(neighbor_i)
                degree_i = decay_factor * KG_i.degree[neighbor_i]
                if num_hop_i == 1:
                    relation_i = node_dict[target_node][i][1]
                else:
                    relation_i = 'degree'

                neighbor_list_degree.append((neighbor_i, relation_i, node_dict[target_node][i][2], degree_i))
                
    if len(neighbor_list_degree) != len(all_neighbor_list):
        raise RuntimeError('length_Error_0')
    
    return neighbor_list_degree, prev_neighbor_degree_list, all_neighbor_list


def entity2text(text_id_l, ent_l_text, ent_ids, rel_ids, graph_id, trans_flg):
    text_all = []
    error_flg = False
#     ent_l_text = ent_ids[entity_id].strip()
#     motif_score_list = []
#     for char_set in text_id_l:
#         motif_score = char_set[1]
#         motif_score_list.append(motif_score)
        
#     motif_score_list_sort = np.array(motif_score_list).argsort()[::-1]
#     text_all_sorted = []
#     for idx in motif_score_list_sort:
#         text_all_sorted.append(text_id_l[idx])

    for char_set in text_id_l:
        triple_flg = char_set[-1]
        head_text = ent_l_text.replace("_", " ")
        rel_text = rel_ids[char_set[1]]
        tail_text = ent_ids[char_set[0]].replace("_", " ")

        translator = Translator()
        word_list = re.findall('[a-zA-Z][^A-Z]*', rel_text)
        if word_list:
            rel_text_en = word_list[0]

            for i in range(len(word_list)):
                if i == 0:
                    continue
                if rel_text_en[-1] == "_" or word_list[i][0] == '-':
                    rel_text_en = rel_text_en + word_list[i]
                else:
                    rel_text_en = rel_text_en + ' ' + word_list[i]

        else:
            try:
                rel_text_en = rel_text
            except:
                print("relation error: ", rel_text)
                error_flg = True
                return None, None, error_flg

        if trans_flg:
            try:
                head_text_en = translator.translate(head_text).text
            except:
                print("head error: ", head_text)
                head_text_en = head_text
                error_flg = True
                return None, None, error_flg
                
            try:
                tail_text_en = translator.translate(tail_text).text
            except:
                print("tail error: ", tail_text)
                tail_text_en = tail_text
                error_flg = True
                return None, None, error_flg
        else:
            head_text_en = head_text
            tail_text_en = tail_text
        
        if char_set[1] == 'degree':
            text_to_add = head_text_en + " " + rel_text_en + " " + tail_text_en
        else:
            if triple_flg == 1:
                text_to_add = head_text_en + " is " + rel_text_en + " of " + tail_text_en
            else:
                text_to_add = tail_text_en + " is " + rel_text_en + " of " + head_text_en
            
        text_all.append(text_to_add)
        
    return text_all, head_text_en, error_flg 


def text_tokenizer(text_all):
    bert_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    tokens = tokenizer.tokenize(text_all)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return tokens, input_ids


decay_factor = 0.5
top_k_motif = int(sys.argv[1])
num_hops = int(sys.argv[2])
KG1 = nx.read_edgelist("KG1")
KG2 = nx.read_edgelist("KG2")
    

ent_ids = {}
ent_ids_cn = {}
rel_ids = {}

with open("ent_ids_1", "r") as ent_1:
    for line in ent_1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        tmp = tmp.replace('_', ' ')
        tmp = tmp.replace("  ", " ")
        content = tmp.split("\t")
        if len(content[0])>5:
            assert False
        text = content[1].split("resource/")[-1]
        ent_ids[content[0]] = text.replace('/', ' ')     
        
with open("ent_ids_2", "r") as ent_2:
    for line in ent_2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        tmp = tmp.replace('_', ' ')
        tmp = tmp.replace("  ", " ")
        content = tmp.split("\t")
        text = content[1].split("resource/")[-1]
        ent_ids[content[0]] = text.replace('/', ' ')

with open("rel_ids_1", "r") as rel_1:
    for line in rel_1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        tmp = tmp.replace('_', ' ')
        tmp = tmp.replace("  ", " ")
        content = tmp.split("\t")
        if len(content[0])>5:
            assert False
        text = content[1].split("/")[-1]
        rel_ids[content[0]] = text

with open("rel_ids_2", "r") as rel_2:
    for line in rel_2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        text = content[1].split("/")[-1]
        rel_ids[content[0]] = text

rel_ids['degree'] = 'has higher order relation with'
        
nodes_tmp_1 = {}
with open("triples_1", "r") as G1:
    for line in G1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        if content[0] in nodes_tmp_1.keys():
            nodes_tmp_1[content[0]].append((content[2], content[1], 0))
        else:
            nodes_tmp_1[content[0]] = [(content[2], content[1], 0)]
            
        if content[2] in nodes_tmp_1.keys():
            nodes_tmp_1[content[2]].append((content[0], content[1], 1))
        else:
            nodes_tmp_1[content[2]] = [(content[0], content[1], 1)]
            

nodes_tmp_2 = {}
with open("triples_2", "r") as G2:
    for line in G2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        if content[0] in nodes_tmp_2.keys():
            nodes_tmp_2[content[0]].append((content[2], content[1], 0))
        else:
            nodes_tmp_2[content[0]] = [(content[2], content[1], 0)]

        if content[2] in nodes_tmp_2.keys():
            nodes_tmp_2[content[2]].append((content[0], content[1], 1))
        else:
            nodes_tmp_2[content[2]] = [(content[0], content[1], 1)]
            
                        
nodes = {}



with open("triples_2", "r") as G2:
    for line in G2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        
        if content[0] not in nodes.keys():
            all_hop_neighbor_list_degree, _, hop_1_neighbor_list = node_degree(content[0], nodes_tmp_2, decay_factor, [], [], KG2, 1)
#             hop_1_neighbor_list = []
#             for set_i in nodes_tmp_2[content[0]]:
#                 neighbor_1_hop = set_i[0]
#                 hop_1_neighbor_list.append(neighbor_1_hop)
            if len(hop_1_neighbor_list) != len(all_hop_neighbor_list_degree):
                raise RuntimeError('length_Error_1')
                
                
            if len(hop_1_neighbor_list) != len(set(hop_1_neighbor_list)):
                raise RuntimeError('neighbor_duplicate_Error_0')
                
            if num_hops > 1:
                hop_1_neighbor_list_new = hop_1_neighbor_list
                hop_2_neighbor_list = []
                for neighbor_1hop_set in nodes_tmp_2[content[0]]:
                    neighbor_1hop = neighbor_1hop_set[0]
                    node_2_hop_neighbor_list_degree_i, all_hop_neighbor_list_degree, hop_2_neighbor_list_i = node_degree(neighbor_1hop, nodes_tmp_2, decay_factor, hop_1_neighbor_list, all_hop_neighbor_list_degree, KG2, 2)
                    all_hop_neighbor_list_degree.extend(node_2_hop_neighbor_list_degree_i)
                    hop_1_neighbor_list_new.extend(hop_2_neighbor_list_i)
                    hop_2_neighbor_list.extend(hop_2_neighbor_list_i)
                    
                    if len(hop_1_neighbor_list) != len(all_hop_neighbor_list_degree):
                        raise RuntimeError('length_Error_2')
                        
                if len(hop_1_neighbor_list_new) != len(set(hop_1_neighbor_list_new)):
                    raise RuntimeError('neighbor_duplicate_Error_1')                    
            if num_hops == 3:
                hop_1_neighbor_list_new_1 = hop_1_neighbor_list_new
                for neighbor_2hop in hop_2_neighbor_list:
                    node_3_hop_neighbor_list_degree_i, all_hop_neighbor_list_degree, hop_3_neighbor_list_i = node_degree(neighbor_2hop, nodes_tmp_2, decay_factor, hop_1_neighbor_list_new, all_hop_neighbor_list_degree, KG2, 3)
                    all_hop_neighbor_list_degree.extend(node_3_hop_neighbor_list_degree_i)
                    hop_1_neighbor_list_new_1.extend(hop_3_neighbor_list_i)
                    
                if len(hop_1_neighbor_list_new_1) != len(set(hop_1_neighbor_list_new_1)):
                    raise RuntimeError('neighbor_duplicate_Error_0')
                    
            neighbor_list_degree_sort = sorted(all_hop_neighbor_list_degree, key=lambda t: t[-1], reverse=True)        
            
            neighbor_len = min(top_k_motif, len(neighbor_list_degree_sort))
            tok_k_neighbor_all_list = []
            for i in range(neighbor_len):
                neighbor_set_i = neighbor_list_degree_sort[i]
                tok_k_neighbor_all_list.append(neighbor_set_i[0])

                if content[0] in nodes.keys():
                    if content[0] != neighbor_set_i[0]:
                        nodes[content[0]].append((neighbor_set_i[0], neighbor_set_i[1], neighbor_set_i[2]))
                else:
                    if content[0] != neighbor_set_i[0]:
                        nodes[content[0]] = [(neighbor_set_i[0], neighbor_set_i[1], neighbor_set_i[2])]
                    
            tok_k_neighbor_all_set = set(tok_k_neighbor_all_list)

            if len(tok_k_neighbor_all_set) != len(tok_k_neighbor_all_list):
                raise RuntimeError('neighbor_duplicate_Error')



with open("triples_1", "r") as G1:
    for line in G1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        
        if content[0] not in nodes.keys():
            all_hop_neighbor_list_degree, _, hop_1_neighbor_list = node_degree(content[0], nodes_tmp_1, decay_factor, [], [], KG1, 1)
#             hop_1_neighbor_list = []
#             for set_i in nodes_tmp_1[content[0]]:
#                 neighbor_1_hop = set_i[0]
#                 hop_1_neighbor_list.append(neighbor_1_hop)
                
                
            if len(hop_1_neighbor_list) != len(all_hop_neighbor_list_degree):
                raise RuntimeError('length_Error_1')
                
                
            if len(hop_1_neighbor_list) != len(set(hop_1_neighbor_list)):
                raise RuntimeError('neighbor_duplicate_Error_0')
            
            if num_hops > 1:
                hop_2_neighbor_list = []
                hop_1_neighbor_list_new = hop_1_neighbor_list
                for neighbor_1hop_set in nodes_tmp_1[content[0]]:
                    neighbor_1hop = neighbor_1hop_set[0]
                    node_2_hop_neighbor_list_degree_i, all_hop_neighbor_list_degree, hop_2_neighbor_list_i = node_degree(neighbor_1hop, nodes_tmp_1, decay_factor, hop_1_neighbor_list, all_hop_neighbor_list_degree, KG1, 2)
                    all_hop_neighbor_list_degree.extend(node_2_hop_neighbor_list_degree_i)
                    hop_1_neighbor_list_new.extend(hop_2_neighbor_list_i)
                    hop_2_neighbor_list.extend(hop_2_neighbor_list_i)
                    
                    if len(hop_1_neighbor_list) != len(all_hop_neighbor_list_degree):
                        raise RuntimeError('length_Error_2')
                        
                if len(hop_1_neighbor_list_new) != len(set(hop_1_neighbor_list_new)):
                    raise RuntimeError('neighbor_duplicate_Error_1')
                    
            if num_hops == 3:
                hop_1_neighbor_list_new_1 = hop_1_neighbor_list_new
                for neighbor_2hop in hop_2_neighbor_list:
                    node_3_hop_neighbor_list_degree_i, all_hop_neighbor_list_degree, hop_3_neighbor_list_i = node_degree(neighbor_2hop, nodes_tmp_1, decay_factor, hop_1_neighbor_list_new, all_hop_neighbor_list_degree, KG1, 3)
                    all_hop_neighbor_list_degree.extend(node_3_hop_neighbor_list_degree_i)
                    hop_1_neighbor_list_new_1.extend(hop_3_neighbor_list_i)
                    
                if len(hop_1_neighbor_list_new_1) != len(set(hop_1_neighbor_list_new_1)):
                    raise RuntimeError('neighbor_duplicate_Error_0')
                
            neighbor_list_degree_sort = sorted(all_hop_neighbor_list_degree, key=lambda t: t[-1], reverse=True)        
            
            neighbor_len = min(top_k_motif, len(neighbor_list_degree_sort))
            
            tok_k_neighbor_all_list = []
            for i in range(neighbor_len):
                neighbor_set_i = neighbor_list_degree_sort[i]
                tok_k_neighbor_all_list.append(neighbor_set_i[0])

                if content[0] in nodes.keys():
                    if content[0] != neighbor_set_i[0]:
                        nodes[content[0]].append((neighbor_set_i[0], neighbor_set_i[1], neighbor_set_i[2]))
                else:
                    if content[0] != neighbor_set_i[0]:
                        nodes[content[0]] = [(neighbor_set_i[0], neighbor_set_i[1], neighbor_set_i[2])]
                    
            tok_k_neighbor_all_set = set(tok_k_neighbor_all_list)

            if len(tok_k_neighbor_all_set) != len(tok_k_neighbor_all_list):
                raise RuntimeError('neighbor_duplicate_Error')
                        



                
### entity correction:
# with open("entity_correction.txt", "r") as ent_correction:
#     for line in ent_correction:
#         tmp = line.strip()
#         tmp = tmp.replace('\n',"")
#         numb = tmp.split("\t")[0].strip()
#         content = tmp.split("\t")[-1].strip()
#         ent_ids[numb] = content
            
            
tokens_len_all = []
tokens_all = []
input_id_len_all = []
input_ids_all = []
text_input_all_1 = []
text_input_all_2 = []
wiki_correction_dict = {}
with open("ill_ent_ids", "r") as GT_file:
    counter = 0
    for line in GT_file:
        counter += 1
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        text_id_l = nodes[content[0]]
        ent_l_text = ent_ids[content[0]]
        text_id_r = nodes[content[1]]
        ent_r_text = ent_ids[content[1]]          
                
        text_to_add_l, head_text_en_l, error_flg_l = entity2text(text_id_l, ent_l_text, ent_ids, rel_ids, "G1: ", False)
        text_to_add_r, head_text_en_r, error_flg_r = entity2text(text_id_r, ent_r_text, ent_ids, rel_ids, "G2: ", False)
        
        if error_flg_l:
            continue
               
        text_input_1 = head_text_en_l + ": "
        text_input_2 = head_text_en_r + ": "
        for text in text_to_add_l:
            text_input_1 = text_input_1 + text + ". "
        for text in text_to_add_r:
            text_input_2 = text_input_2 + text + ". "
        
        text_input_1 = text_input_1.replace("  ", " ")
        text_input_1 = text_input_1.replace(" .", ".")
        text_input_1 = text_input_1.replace(" )", ")")
        text_input_2 = text_input_2.replace("  ", " ")
        text_input_2 = text_input_2.replace(" .", ".")
        text_input_2 = text_input_2.replace(" )", ")")
        text_input_1 = text_input_1.strip()
        text_input_2 = text_input_2.strip()
        text_input_all_1.append(text_input_1)
        text_input_all_2.append(text_input_2)


        if counter % 20 == 0:
            text_save("data/text_input_no_train_11_wxt_"+str(sys.argv[2])+"_hop_degree_"+str(sys.argv[1])+".txt", text_input_all_1)
            text_save("data/text_input_no_train_22_wxt_"+str(sys.argv[2])+"_hop_degree_"+str(sys.argv[1])+".txt", text_input_all_2)

text_save("data/text_input_no_train_11_wxt_"+str(sys.argv[2])+"_hop_degree_"+str(sys.argv[1])+".txt", text_input_all_1)
text_save("data/text_input_no_train_22_wxt_"+str(sys.argv[2])+"_hop_degree_"+str(sys.argv[1])+".txt", text_input_all_2)



        