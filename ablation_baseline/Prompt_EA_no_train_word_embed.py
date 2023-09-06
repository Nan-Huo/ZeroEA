import numpy as np 
import os
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from scipy import spatial
import scipy


def get_hits(Lvec, Rvec, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='euclidean')
    top_lr = [0] * len(top_k)
    RR_left = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        RR_left += 1/(rank_index+1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    RR_right = 0
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        RR_right += 1/(rank_index+1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_left / Lvec.shape[0] * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / Lvec.shape[0] * 100))
    print('MRR: %.2f%%' % (RR_right / Lvec.shape[0] * 100))


def calc_cos_sim(vec1,vec2):
    cos_sim = 1 - spatial.distance.cosine(vec1, vec2)
    
    return cos_sim


def get_target_embed(filename, tokenizer, model):
    with open(filename, "r") as input_f:
        target_embed = []
        counter = 0
        counter_list = []
        for line in input_f:
            tmp = line.strip()
            real_label = tmp.replace("\n", '')
            
            
            ### remove kuohao
#             while True:
#                 if "(" in real_label:
#                     try:
#                         del_idx_1 = real_label.index("(")
#                         del_idx_2 = real_label.index(")")
#                         real_label = real_label[:del_idx_1] + real_label[del_idx_2+1:]
#                     except:
#                         break
#                 else:
#                     break
                 
            
            entity_text = real_label.split(":")[0].strip()
            
            tokens = tokenizer(
                entity_text,                     # 分词文本
                return_token_type_ids=False,    # 返回是前一句还是后一句
                return_attention_mask=True,     # 返回attention_mask
                return_tensors='pt')            # 返回pytorch类型的tensor
            
            tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

            with torch.no_grad():    #将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
                encoded_layers = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])['last_hidden_state']
                last_hidden_state = encoded_layers[0]
                cls_idx = tokenized_text.index("[CLS]")
                summed_embed = last_hidden_state[cls_idx]
                tokens = summed_embed
                target_embed.append(tokens.numpy())

            if counter > thresh_num:
                break
            counter += 1
            
    return target_embed
    
thresh_num = 15000
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()  #这里是验证模型，可以节省很多不必要的反向传播

target_embed_1 = get_target_embed('text_input_no_train_11_wxt.txt', tokenizer, model)
target_embed_2 = get_target_embed('text_input_no_train_22_wxt.txt', tokenizer, model)

Lvec = np.array(target_embed_1)
Rvec = np.array(target_embed_2)
get_hits(Lvec, Rvec)
