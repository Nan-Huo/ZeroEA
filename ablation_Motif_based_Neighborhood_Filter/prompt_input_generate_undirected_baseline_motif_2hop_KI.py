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
    

def smart_traingle_adjacency_matrix(G: nx.Graph):
    A = nx.to_numpy_array(G)
    A2 = np.dot(A, A)
    T = np.multiply(A2, A)
    np.fill_diagonal(T, 0)
    return T
    
def text_save(filename, data):
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i])#.replace('[','').replace(']','')
        s = s + "\n"
        file.write(s)
    file.close()


def entity2text_2hop(entity_id, ent_ids, rel_ids, trans_flg):
    text_all = []
    error_flg = False
    text_id_l = nodes[entity_id]
    for char in text_id_l:
        triple_flg = char_set[1]
        char = char_set[0]
        content = char.split("\t")
        head_text = ent_l_text.replace("_", " ")
        rel_text = rel_ids[content[0]]
        tail_text = ent_ids[content[1]].replace("_", " ")

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

        if triple_flg == 1:
            text_to_add = head_text_en + " is " + rel_text_en + " of " + tail_text_en
        else:
            text_to_add = tail_text_en + " is " + rel_text_en + " of " + head_text_en
        
        text_all.append(text_to_add)
        
    return text_all, head_text_en, error_flg


def entity2text(text_id_l, entity_id, ent_l_text, ent_ids, rel_ids, nodes, graph_id, trans_flg):
    text_all = []
    error_flg = False
#     ent_l_text = ent_ids[entity_id].strip()
    motif_score_list = []
    for char_set in text_id_l:
        motif_score = char_set[1]
        motif_score_list.append(motif_score)
        
    motif_score_list_sort = np.array(motif_score_list).argsort()[::-1]
    text_all_sorted = []
    for idx in motif_score_list_sort:
        text_all_sorted.append(text_id_l[idx])

    for char_set in text_all_sorted:
        triple_flg = char_set[-1]
        char = char_set[0]
        content = char.split("\t")
        head_text = ent_l_text.replace("_", " ")
        rel_text = rel_ids[content[0]]
        tail_text = ent_ids[content[1]].replace("_", " ")

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
        
        if triple_flg == 1:
            text_to_add = head_text_en + " is " + rel_text_en + " of " + tail_text_en
        else:
            text_to_add = tail_text_en + " is " + rel_text_en + " of " + head_text_en
#         text_2hop,_,_ = entity2text_2hop(content[1], ent_ids, rel_ids, False)
        text_all.append(text_to_add)
#         text_all.extend(text_2hop)
        
    return text_all, head_text_en, error_flg 


def text_tokenizer(text_all):
    bert_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    tokens = tokenizer.tokenize(text_all)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return tokens, input_ids


top_k_motif = int(sys.argv[1])
KG1 = nx.read_edgelist("KG1")
KG2 = nx.read_edgelist("KG2")
KG1_adjacency_matrix_idx = {}
KG1_adjacency_matrix_idx_counter = {}
for idx,node in enumerate(list(KG1.nodes())):
    KG1_adjacency_matrix_idx[str(node)] = idx
    KG1_adjacency_matrix_idx_counter[idx] = str(node)
    
KG2_adjacency_matrix_idx = {}
KG2_adjacency_matrix_idx_counter = {}
for idx,node in enumerate(list(KG2.nodes())):
    KG2_adjacency_matrix_idx[str(node)] = idx
    KG2_adjacency_matrix_idx_counter[idx] = str(node)

KG1_adjacency_matrix = smart_traingle_adjacency_matrix(KG1)
KG2_adjacency_matrix = smart_traingle_adjacency_matrix(KG2)

KG1_adjacency_matrix_2hop = np.dot(KG1_adjacency_matrix, KG1_adjacency_matrix)
KG2_adjacency_matrix_2hop = np.dot(KG2_adjacency_matrix, KG2_adjacency_matrix)

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

rel_ids['motif'] = 'MOTIF relation'
        
nodes_tmp_1 = {}
with open("triples_1", "r") as G1:
    for line in G1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        if content[0] in nodes_tmp_1.keys():
            nodes_tmp_1[content[0]].append((content[2], content[1]+"\t"+content[2], 0))
        else:
            nodes_tmp_1[content[0]] = [(content[2], content[1]+"\t"+content[2], 0)]
            
        if content[2] in nodes_tmp_1.keys():
            nodes_tmp_1[content[2]].append((content[0], content[1]+"\t"+content[0], 1))
        else:
            nodes_tmp_1[content[2]] = [(content[0], content[1]+"\t"+content[0], 1)]
            

nodes_tmp_2 = {}
with open("triples_2", "r") as G2:
    for line in G2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        if content[0] in nodes_tmp_2.keys():
            nodes_tmp_2[content[0]].append((content[2], content[1]+"\t"+content[2], 0))
        else:
            nodes_tmp_2[content[0]] = [(content[2], content[1]+"\t"+content[2], 0)]

        if content[2] in nodes_tmp_2.keys():
            nodes_tmp_2[content[2]].append((content[0], content[1]+"\t"+content[0], 1))
        else:
            nodes_tmp_2[content[2]] = [(content[0], content[1]+"\t"+content[0], 1)]
            
        
nodes = {}
with open("triples_1", "r") as G1:
    for line in G1:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        
        motif_idx_1 = KG1_adjacency_matrix_idx[content[0]]
        motif_idx_2 = KG1_adjacency_matrix_idx[content[2]]
        if content[0] not in nodes.keys():
            neighbor_list = []
            rel_nei_list = []
            direction_flg_list = []
            for neighbor_set in nodes_tmp_1[content[0]]:
                neighbor_list.append(neighbor_set[0])
                rel_nei_list.append(neighbor_set[1])
                direction_flg_list.append(neighbor_set[2])             
                        
            counter_top = 0
            motif_share_list = KG1_adjacency_matrix[motif_idx_1, :]
            motif_share_list_sort = motif_share_list.argsort()[::-1]
            motif_share_list_2hop = KG1_adjacency_matrix_2hop[motif_idx_1, :]
            motif_share_list_sort_2hop = motif_share_list_2hop.argsort()[::-1]
            for motif_idx_i in motif_share_list_sort:
                neighbor_candidate = KG1_adjacency_matrix_idx_counter[motif_idx_i]
                if motif_idx_i == motif_idx_1:
                    continue
                if motif_share_list[motif_idx_i] > 0:
                    counter_top += 1  
                    if neighbor_candidate in neighbor_list:
                        idx_rel_nei = neighbor_list.index(neighbor_candidate)
                        rel_neighbor = rel_nei_list[idx_rel_nei]
                        direction_flg = direction_flg_list[idx_rel_nei]

                        if content[0] in nodes.keys():
                            nodes[content[0]].append((rel_neighbor, motif_share_list[motif_idx_i], direction_flg))
                        else:
                            nodes[content[0]] = [(rel_neighbor, motif_share_list[motif_idx_i], direction_flg)]
                    else:
                        raise RuntimeError('Motif_Neighbor_Error')
                        
                if motif_share_list[motif_idx_i] == 0 or counter_top >= top_k_motif:
                    break
                    
            if counter_top < top_k_motif:
                for motif_idx_i in motif_share_list_sort_2hop:
                    neighbor_candidate = KG1_adjacency_matrix_idx_counter[motif_idx_i]
                    if motif_idx_i == motif_idx_1:
                        continue
                    if motif_share_list_2hop[motif_idx_i] > 0:
                        if neighbor_candidate not in neighbor_list:
                            counter_top += 1

                            if content[0] in nodes.keys():
                                nodes[content[0]].append(("motif\t"+neighbor_candidate, motif_share_list_2hop[motif_idx_i], 1))
                            else:
                                nodes[content[0]] = [("motif\t"+neighbor_candidate, motif_share_list_2hop[motif_idx_i], 1)]

                    if motif_share_list_2hop[motif_idx_i] == 0 or counter_top >= top_k_motif:
                        break
                        
            if counter_top <= 2:
                for item in nodes_tmp_1[content[0]]:
                    counter_top += 1
                    if content[0] in nodes.keys():
                        nodes[content[0]].append((item[1], 0, item[-1]))
                    else:
                        nodes[content[0]] = [(item[1], 0, item[-1])]
                        
                    if counter_top >= top_k_motif:
                        break
                        

with open("triples_2", "r") as G2:
    for line in G2:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        content = tmp.split("\t")
        motif_idx_1 = KG2_adjacency_matrix_idx[content[0]]
        motif_idx_2 = KG2_adjacency_matrix_idx[content[2]]
        if content[0] not in nodes.keys():
            neighbor_list = []
            rel_nei_list = []
            direction_flg_list = []
            for neighbor_set in nodes_tmp_2[content[0]]:
                neighbor_list.append(neighbor_set[0])
                rel_nei_list.append(neighbor_set[1])
                direction_flg_list.append(neighbor_set[2])             
                        
            counter_top = 0
            motif_share_list = KG2_adjacency_matrix[motif_idx_1, :]
            motif_share_list_sort = motif_share_list.argsort()[::-1]
            motif_share_list_2hop = KG2_adjacency_matrix_2hop[motif_idx_1, :]
            motif_share_list_sort_2hop = motif_share_list_2hop.argsort()[::-1]
            for motif_idx_i in motif_share_list_sort:
                neighbor_candidate = KG2_adjacency_matrix_idx_counter[motif_idx_i]
                if motif_idx_i == motif_idx_1:
                    continue
                if motif_share_list[motif_idx_i] > 0:
                    counter_top += 1  
                    if neighbor_candidate in neighbor_list:
                        idx_rel_nei = neighbor_list.index(neighbor_candidate)
                        rel_neighbor = rel_nei_list[idx_rel_nei]
                        direction_flg = direction_flg_list[idx_rel_nei]

                        if content[0] in nodes.keys():
                            nodes[content[0]].append((rel_neighbor, motif_share_list[motif_idx_i], direction_flg))
                        else:
                            nodes[content[0]] = [(rel_neighbor, motif_share_list[motif_idx_i], direction_flg)]
                        
                if motif_share_list[motif_idx_i] == 0 or counter_top >= top_k_motif:
                    break
                    
            if counter_top < top_k_motif:
                for motif_idx_i in motif_share_list_sort_2hop:
                    neighbor_candidate = KG2_adjacency_matrix_idx_counter[motif_idx_i]
                    if motif_idx_i == motif_idx_1:
                        continue
                    if motif_share_list_2hop[motif_idx_i] > 0:
                        if neighbor_candidate not in neighbor_list:
                            counter_top += 1

                            if content[0] in nodes.keys():
                                nodes[content[0]].append(("motif\t"+neighbor_candidate, motif_share_list_2hop[motif_idx_i], 1))
                            else:
                                nodes[content[0]] = [("motif\t"+neighbor_candidate, motif_share_list_2hop[motif_idx_i], 1)]

                    if motif_share_list_2hop[motif_idx_i] == 0 or counter_top >= top_k_motif:
                        break
                        
            if counter_top <= 2:
                for item in nodes_tmp_2[content[0]]:
                    counter_top += 1
                    if content[0] in nodes.keys():
                        nodes[content[0]].append((item[1], 0, item[-1]))
                    else:
                        nodes[content[0]] = [(item[1], 0, item[-1])]
                        
                    if counter_top >= top_k_motif:
                        break

                
### entity correction:
with open("entity_correction.txt", "r") as ent_correction:
    for line in ent_correction:
        tmp = line.strip()
        tmp = tmp.replace('\n',"")
        numb = tmp.split("\t")[0].strip()
        content = tmp.split("\t")[-1].strip()
        ent_ids[numb] = content
            
            
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
                
        text_to_add_l, head_text_en_l, error_flg_l = entity2text(text_id_l, content[0], ent_l_text, ent_ids, rel_ids, nodes, "G1: ", False)
        text_to_add_r, head_text_en_r, error_flg_r = entity2text(text_id_r, content[1], ent_r_text, ent_ids, rel_ids, nodes, "G2: ", False)
        
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
            text_save("data/text_input_no_train_11_wxt_2hop_motif_KI_"+str(sys.argv[1])+".txt", text_input_all_1)
            text_save("data/text_input_no_train_22_wxt_2hop_motif_KI_"+str(sys.argv[1])+".txt", text_input_all_2)

text_save("data/text_input_no_train_11_wxt_2hop_motif_KI_"+str(sys.argv[1])+".txt", text_input_all_1)
text_save("data/text_input_no_train_22_wxt_2hop_motif_KI_"+str(sys.argv[1])+".txt", text_input_all_2)





        