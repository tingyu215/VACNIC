import os
from re import I
from charset_normalizer import detect
import pymongo
from pymongo import MongoClient
import numpy as np
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# from torchtext.data import get_tokenizer
from nltk.tokenize import word_tokenize
from transformers import BartTokenizer, BartModel
import json
from torchvision import transforms
import spacy
import copy
import unidecode


def collate_fn_goodnews_entity_type(batch):
    # print(len(batch))
    article_list = []
    article_id_list = []
    article_ner_mask_id_list = []

    caption_list = []
    caption_id_list = []
    caption_id_clip_list = []
    names_art_list = []
    names_art_ids_list = []
    org_norp_gpe_loc_art_list = []
    org_norp_gpe_loc_art_ids_list = []
    names_list = []
    names_ids_list = []
    org_norp_gpe_loc_list = []
    org_norp_gpe_loc_ids_list = []

    names_ids_flatten_list = []
    org_norp_gpe_loc_ids_flatten_list = []

    all_gt_ner_list = []
    all_gt_ner_ids_list = []
    face_emb_list = []
    obj_emb_list = []
    img_tensor_list = []
    face_pad = torch.ones((1, 512))
    obj_pad = torch.ones((1,2048))

    person_id_positions_list = []
    person_id_positions_cap_list = []

    for i in range(len(batch)):
        article, article_ids,article_ner_mask_ids, caption, caption_ids, caption_ids_clip, names_art, org_norp_gpe_loc_art, names_art_ids, org_norp_gpe_loc_art_ids, names, org_norp_gpe_loc, names_ids, org_norp_gpe_loc_ids, all_gt_ner_ids, all_gt_ner, face_emb, obj_emb, img_tensor, person_id_positions, person_id_positions_cap = batch[i]["article"], batch[i]["article_ids"], batch[i]["article_ner_mask_ids"],batch[i]["caption"], batch[i]["caption_ids"], batch[i]["caption_ids_clip"], batch[i]["names_art"], batch[i]["org_norp_gpe_loc_art"], batch[i]["names_art_ids"], batch[i]["org_norp_gpe_loc_art_ids"], batch[i]["names"], batch[i]["org_norp_gpe_loc"], batch[i]["names_ids"], batch[i]["org_norp_gpe_loc_ids"], batch[i]["all_gt_ner_ids"], batch[i]["all_gt_ner"], batch[i]["face_emb"], batch[i]["obj_emb"], batch[i]["img_tensor"], batch[i]["person_id_positions"], batch[i]["person_id_positions_cap"]

        names_ids_flatten, org_norp_gpe_loc_ids_flatten = batch[i]["names_ids_flatten"], batch[i]["org_norp_gpe_loc_ids_flatten"]

        names_ids_flatten_list.append(names_ids_flatten)
        org_norp_gpe_loc_ids_flatten_list.append(org_norp_gpe_loc_ids_flatten)

        article_list.append(article)
        article_id_list.append(article_ids)
        article_ner_mask_id_list.append(article_ner_mask_ids)
        caption_list.append(caption)
        caption_id_list.append(caption_ids)
        if caption_ids_clip is not None:
            caption_id_clip_list.append(caption_ids_clip)

        names_art_list.append(names_art)
        names_art_ids_list.append(names_art_ids)
        org_norp_gpe_loc_art_list.append(org_norp_gpe_loc_art)
        org_norp_gpe_loc_art_ids_list.append(org_norp_gpe_loc_art_ids)

        names_list.append(names)
        names_ids_list.append(names_ids)
        org_norp_gpe_loc_list.append(org_norp_gpe_loc)
        org_norp_gpe_loc_ids_list.append(org_norp_gpe_loc_ids)

        all_gt_ner_list.append(all_gt_ner)
        all_gt_ner_ids_list.append(all_gt_ner_ids)
        
        face_emb_list.append(face_emb)
        obj_emb_list.append(obj_emb)
        
        img_tensor_list.append(img_tensor)
        
        person_id_positions_list.append(person_id_positions)
        person_id_positions_cap_list.append(person_id_positions_cap)
    
    max_len_input = get_max_len([article_id_list, article_ner_mask_id_list])
    # article_ids_batch = pad_article(article_id_list, 1)
    article_ids_batch = pad_sequence(article_id_list, 1, max_len=max_len_input)
    article_ner_mask_ids_batch = pad_sequence(article_ner_mask_id_list, 1, max_len=max_len_input)
    # print(f"article batch: {article_ids_batch.size()}")
    caption_ids_batch = pad_sequence(caption_id_list, 1)
    # print(caption_ids_batch.size())
    if len(caption_id_clip_list) > 0:
        caption_ids_clip_batch = pad_sequence(caption_id_clip_list, 0)
    else:
        caption_ids_clip_batch = torch.empty((1,1))
    
    max_len_art_ids = get_max_len([names_art_ids_list, org_norp_gpe_loc_art_ids_list])
    
    max_len_name_ids = get_max_len_list(names_ids_list)
    max_len_org_norp_gpe_loc_ids = get_max_len_list(org_norp_gpe_loc_ids_list)
    
    names_art_ids_batch = pad_sequence(names_art_ids_list, 1, max_len=max_len_art_ids)
    org_norp_gpe_loc_art_ids_batch = pad_sequence(org_norp_gpe_loc_art_ids_list, 1, max_len=max_len_art_ids)

    names_ids_batch = pad_sequence_from_list(names_ids_list, special_token_id=50266, bos_token_id=0, pad_token_id=1, eos_token_id=2,  max_len=max_len_name_ids)

    org_norp_gpe_loc_ids_batch = pad_sequence_from_list(org_norp_gpe_loc_ids_list, special_token_id=50266, bos_token_id=0, pad_token_id=1, eos_token_id=2, max_len=max_len_org_norp_gpe_loc_ids)

    all_gt_ner_ids_batch = pad_sequence(all_gt_ner_ids_list, 1)

    max_len_ids_flatten = get_max_len([names_ids_flatten_list, org_norp_gpe_loc_ids_flatten_list])
    names_ids_flatten_batch = pad_sequence(names_ids_flatten_list, 1, max_len=max_len_ids_flatten)
    org_norp_gpe_loc_ids_flatten_batch = pad_sequence(org_norp_gpe_loc_ids_flatten_list, 1, max_len=max_len_ids_flatten)
    
    img_batch = torch.stack(img_tensor_list, dim=0).squeeze(1)
    # print(img_batch.size())

    face_batch = pad_tensor_feat(face_emb_list, face_pad)
    obj_batch = pad_tensor_feat(obj_emb_list, obj_pad)

    return {"article": article_list, "article_ids":article_ids_batch.squeeze(1), "article_ner_mask_ids":article_ner_mask_ids_batch.squeeze(1), "caption": caption_list, "caption_ids": caption_ids_batch.squeeze(1), "caption_ids_clip": caption_ids_clip_batch.squeeze(1), "names_art": names_art_list, "names_art_ids": names_art_ids_batch.squeeze(1), "org_norp_gpe_loc_art": org_norp_gpe_loc_art_list, "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids_batch.squeeze(1), "names":names_list, "names_ids":names_ids_batch.squeeze(1), "org_norp_gpe_loc":org_norp_gpe_loc_list, "org_norp_gpe_loc_ids":org_norp_gpe_loc_ids_batch.squeeze(1), "all_gt_ner_ids":all_gt_ner_ids_batch.squeeze(1), "all_gt_ner":all_gt_ner_list, "face_emb":face_batch.float(), "obj_emb":obj_batch.float(), "img_tensor":img_batch, "person_id_positions":person_id_positions_list, "person_id_positions_cap":person_id_positions_cap_list, "names_ids_flatten":names_ids_flatten_batch.squeeze(1), "org_norp_gpe_loc_ids_flatten":org_norp_gpe_loc_ids_flatten_batch.squeeze(1)}



def get_max_len_list(seq_list_of_list):
    # input list of seq_list, output max len
    max_len_list = []
    for seq_list in seq_list_of_list:
        # print(seq_list)
        max_len_list.extend([len(seq) for seq in seq_list])
    return max(max_len_list)


# def pad_sequence_from_list(seq_list_list, pad_token_id, bos_token_id, eos_token_id, special_token_id_list, max_len):
#     max_num_seq = max([len(seq_list) for seq_list in seq_list_list])
#     # print(seq_list_list)
#     if max_len < 3:
#         # NONAME and NOORG takes 3 tokens
#         max_len = 3
#     padded_list_all = []
#     for seq_list in seq_list_list:
#         padded_list = []
#         for seq in seq_list:
#             # pad in each sample
#             pad_num = max_len - len(seq)
#             seq = [bos_token_id] + seq + [eos_token_id] + [pad_token_id] * pad_num
#             # print(seq, pad_num)
#             if max_num_seq == 1:
#                 padded_list.append([seq])
#             else:
#                 padded_list.append(seq)
#         if len(seq_list) < max_num_seq:
#             # pad in each batch
#             pad_batch_wise = [bos_token_id] + special_token_id_list + [eos_token_id] + [pad_token_id] * (max_len-3)
#             for i in range(max_num_seq - len(seq_list)):
#                 padded_list.append(pad_batch_wise)
#         padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
#     # return torch.tensor(padded_list_all, dtype=torch.long)
#     # print(padded_list_all)
#     return torch.stack(padded_list_all)


def pad_sequence_from_list(seq_list_list, special_token_id, bos_token_id, pad_token_id, eos_token_id, max_len):
    # special_token_id: <NONAME>
    max_num_seq = max([len(seq_list) for seq_list in seq_list_list])
    padded_list_all = []
    for seq_list in seq_list_list:
        padded_list = []
        for seq in seq_list:
            # pad in each sample
            pad_num = max_len - len(seq)
            seq = seq + [pad_token_id] * pad_num
            # print(seq, pad_num)
            if max_num_seq == 1:
                padded_list.append([seq])
            else:
                padded_list.append(seq)
        if len(seq_list) < max_num_seq:
            # pad in each batch
            pad_batch_wise = [bos_token_id] + [special_token_id] + [eos_token_id] + [pad_token_id] * (max_len-3)
            for i in range(max_num_seq - len(seq_list)):
                padded_list.append(pad_batch_wise)
        padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
    return torch.stack(padded_list_all)


def get_max_len(seq_tensor_list_of_list):
    # input list of seq_tensor_list, output max len
    max_len_list = []
    for seq_tensor_list in seq_tensor_list_of_list:
        max_len_list.append(max([seq.size(1) for seq in seq_tensor_list]))
    return max(max_len_list)


def pad_sequence(seq_tensor_list, pad_token_id, max_len=None):
    if max_len is None:
        max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    for seq in seq_tensor_list:
        # print(seq.size())
        pad_num = max_len - seq.size(1)
        if pad_num > 0:
            to_be_padded = torch.tensor([pad_token]*pad_num, dtype=torch.long).unsqueeze(0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            padded_list.append(torch.cat((seq, to_be_padded), dim=1))
        else:
            padded_list.append(seq)
    return torch.stack(padded_list)


def pad_sequence_lm(seq_tensor_list, sos_token_id, eos_token_id, pad_token_id):
    max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    sos_token = torch.tensor([sos_token_id], dtype=torch.long).unsqueeze(0)
    eos_token = torch.tensor([eos_token_id], dtype=torch.long).unsqueeze(0)
    for seq in seq_tensor_list:
        # print(seq.size())
        pad_num = max_len + 2 - seq.size(1)
        if pad_num > 0:
            to_be_padded = torch.tensor([pad_token]*pad_num, dtype=torch.long).unsqueeze(0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            padded_list.append(torch.cat((sos_token, seq, eos_token, to_be_padded), dim=1))
        else:
            padded_list.append(torch.cat((sos_token, seq, eos_token), dim=1))
    return torch.stack(padded_list)



def pad_article(seq_tensor_list, pad_token_id):
    """2D padding to pad list of sentences"""
    max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    for seq in seq_tensor_list:
        # print(seq.size())
        pad_num = max_len - seq.size(1)
        if pad_num > 0:
            to_be_padded = seq.size(0)*[torch.tensor([pad_token]*pad_num, dtype=torch.long)]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            # print(to_be_padded.size(), seq.size())
            padded_list.append(torch.cat((seq, to_be_padded), dim=1))
        else:
            padded_list.append(seq)
    
    max_len_article = max([seq.size(0) for seq in seq_tensor_list])
    padded_list_out = []
    for seq in padded_list:
        # print(seq.size())
        pad_num = max_len_article - seq.size(0)
        if pad_num > 0:
            to_be_padded = pad_num * [torch.tensor([pad_token]*max_len, dtype=torch.long)]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            padded_list_out.append(torch.cat((seq, to_be_padded), dim=0))
        else:
            padded_list_out.append(seq)
    return torch.stack(padded_list_out)


def pad_tensor_feat(feat_np_list, pad_feat_tensor):
    # tensor_list = [torch.from_numpy(feat) for feat in feat_np_list]
    len_list = []
    for feat in feat_np_list:
        if feat.shape[1] == 0:
            len_list.append(0)
        else:
            len_list.append(feat.shape[0])
    max_len = max(len_list)
    # print(max_len)
    padded_list = []
    for i, feat in enumerate(feat_np_list):
        pad_num = max_len - len_list[i]
        if pad_num > 0:
            to_be_padded = pad_num* [pad_feat_tensor]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            to_be_padded = to_be_padded.squeeze(1)
            # print(to_be_padded.size())
            if feat.shape[1] != 0:
                # padded_list.append(torch.stack((torch.from_numpy(feat), to_be_padded), dim=0).squeeze(1))
                padded_list.append(torch.cat((torch.from_numpy(feat), to_be_padded), dim=0).squeeze(1))
            else:
                padded_list.append(to_be_padded)
        elif max_len == 0:
            to_be_padded = 1* [pad_feat_tensor]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            to_be_padded = to_be_padded.squeeze(1)
            padded_list.append(to_be_padded)
        else:
            # print(torch.from_numpy(feat).size())
            padded_list.append(torch.from_numpy(feat))
    return  torch.stack(padded_list)



def make_new_entity_ids(caption, ent_list, tokenizer, ent_separator="<ent>", max_length=80):
    caption_ids_ner = tokenizer(caption, add_special_tokens=False)["input_ids"]
    # print(caption_ids_ner)

    sep_token = tokenizer(ent_separator, add_special_tokens=False)["input_ids"]
    # print(sep_token)

    noname_token = tokenizer("<NONAME>")["input_ids"][1:-1]

    ent_ids_flatten = []
    ent_ids_separate = []

    for ent in ent_list:
        # in case entities were in the middle of the sentence
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # print(ent_ids_original)
        # if ent_ids_original in article_ids_ner:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # print(ent_ids_original)
            ent_ids_flatten.extend(ent_ids_original)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break
            else:
                continue
        else:
            # print(ent_ids_original)
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            # print(f"start:{ent_ids_original_start}")
            ent_ids_flatten.extend(ent_ids_original_start)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original_start + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break
            else:
                continue
    if len(ent_ids_flatten) ==0:
        ent_ids_flatten.extend(noname_token)

    # torch.LongTensor([[tokenizer.bos_token_id] + ent_ids_flatten + [tokenizer.eos_token_id]])
    ent_ids_flatten = [tokenizer.bos_token_id] + ent_ids_flatten + [tokenizer.eos_token_id]
    if len(ent_ids_flatten) < max_length:
        ent_ids_flatten = ent_ids_flatten + [tokenizer.pad_token_id] * (max_length -  len(ent_ids_flatten))
    
    ent_ids_separate.append([tokenizer.bos_token_id] + noname_token + [tokenizer.eos_token_id])
    ent_ids_separate = pad_list(ent_ids_separate, tokenizer.pad_token_id)
    return torch.LongTensor([ent_ids_flatten]), ent_ids_separate
 

def pad_list(list_of_name_ids, pad_token):
    max_len = max([len(seq) for seq in list_of_name_ids])
    padded_list = []
    for seq in list_of_name_ids:
        if len(seq) == max_len:
            padded_list.append(seq)
        else:
            padded_num = max_len - len(seq)
            seq.extend([pad_token] * padded_num)
            padded_list.append(seq)
    return padded_list


def get_person_ids_position(article_ids_replaced, person_token_id=50265, article_max_length=512, is_tgt_input=False):
    position_list = []
    # for i in range(len(article_ids_replaced)):
    i = 0
    while i < len(article_ids_replaced):
        position_i = []
        if article_ids_replaced[i] == person_token_id and i < article_max_length:
            if is_tgt_input:
                position_i.append(i+1)
            else:
                position_i.append(i)
            for j in range(i, len(article_ids_replaced)):
                if article_ids_replaced[j] == person_token_id:
                    continue
                else:
                    if is_tgt_input:
                        position_i.append(j)
                    else:
                        position_i.append(j-1)
                    i=j-1
                    # print(i)
                    break
            position_list.append(position_i)
            # print("i:",i)
        i += 1
    return position_list


class GoodNewsDictDatasetEntityType(Dataset):
    def __init__(self, data_dict, data_base_dir, tokenizer, use_clip_tokenizer=False, entity_token_start="no", entity_token_end="no", transform=None, max_article_len=512, max_ner_type_len=128):
        super().__init__()
        self.data_dict = copy.deepcopy(data_dict)
        self.face_dir = os.path.join(data_base_dir, "faces")
        self.obj_dir = os.path.join(data_base_dir, "objects")
        self.article_dir = os.path.join(data_base_dir, "articles")
        self.article_ner_mask_dir = os.path.join(data_base_dir, "articles_full_pog_by_count")
        self.img_dir = os.path.join(data_base_dir, "images_processed")
        self.tokenizer = tokenizer
        self.use_clip_tokenizer = use_clip_tokenizer
        self.max_len = max_article_len
        self.transform = transform
        self.entity_token_start = entity_token_start
        self.entity_token_end = entity_token_end
        self.hash_ids = [*data_dict.keys()]
        self.max_ner_type_len = max_ner_type_len
    def __getitem__(self, index):
        hash_id = self.hash_ids[index]
        img = Image.open(os.path.join(self.img_dir, f"{hash_id}.jpg")).convert('RGB')
        if self.data_dict[hash_id]["face_emb_dir"] != []:
            face_emb = np.load(os.path.join(self.face_dir, f"{hash_id}.npy"))
            names = self.data_dict[hash_id]["names"]
        else:
            face_emb = np.array([[]])
            names = []
        
        if self.data_dict[hash_id]["obj_emb_dir"] != []:
            obj_emb = np.load(os.path.join(self.obj_dir, f"{hash_id}.npy"))
        else:
            obj_emb = np.array([[]])
        
        with open(os.path.join(os.path.join(self.article_dir, f"{hash_id}.txt"))) as f:
            # article = f.readlines()
            article = f.read()
        
        caption = self.data_dict[hash_id]["caption"]
        names = self.data_dict[hash_id]["names"]
        org_norp = self.data_dict[hash_id]["org_norp"]
        gpe_loc = self.data_dict[hash_id]["gpe_loc"]
        names_art = self.data_dict[hash_id]["names_art"]
        org_norp_art = self.data_dict[hash_id]["org_norp_art"]
        gpe_loc_art = self.data_dict[hash_id]["gpe_loc_art"]
        # article = preprocess_article(article)

        # delete repetitive names
        new_names_art_list = []
        for i in range(len(names_art)):
            if compare_ner(names_art[i], names_art[:i] + names_art[i+1:]):
                continue
            else:
                new_names_art_list.append(names_art[i])
        concat_names_art = concat_ner(new_names_art_list, self.entity_token_start, self.entity_token_end)
        concat_names = concat_ner(names, self.entity_token_start, self.entity_token_end)


        # delete repetitive orgs
        new_org_norp_art_list = []
        for i in range(len(org_norp_art)):
            if compare_ner(org_norp_art[i], org_norp_art[:i] + org_norp_art[i+1:]):
                continue
            else:
                new_org_norp_art_list.append(org_norp_art[i])
        concat_org_norp_art = concat_ner(new_org_norp_art_list, self.entity_token_start, self.entity_token_end)
        concat_org_norp = concat_ner(org_norp, self.entity_token_start, self.entity_token_end)


        # delete repetitive gpe
        new_gpe_loc_art_list = []
        for i in range(len(gpe_loc_art)):
            if compare_ner(gpe_loc_art[i], gpe_loc_art[:i] + gpe_loc_art[i+1:]):
                continue
            else:
                new_gpe_loc_art_list.append(gpe_loc_art[i])
        concat_gpe_loc_art = concat_ner(new_gpe_loc_art_list, self.entity_token_start, self.entity_token_end)
        concat_gpe_loc = concat_ner(gpe_loc, self.entity_token_start, self.entity_token_end)

        all_gt_ner = names + org_norp + gpe_loc
        # print(all_gt_ner)
        concat_gt_ner = concat_ner(all_gt_ner, self.entity_token_start, self.entity_token_end)
        # print(concat_gt_ner)
        gt_ner_ids = self.tokenizer(concat_gt_ner, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        
        article_ids = self.tokenizer(article, return_tensors="pt", truncation=True, padding=True,max_length=self.max_len)["input_ids"]

        with open(os.path.join(os.path.join(self.article_ner_mask_dir, f"{hash_id}.json"))) as f:
            article_ner_mask_dict = json.load(f)
        if len(article_ner_mask_dict["input_ids"]) > self.max_len:
            article_ner_mask_ids = torch.LongTensor(article_ner_mask_dict["input_ids"][:self.max_len-1] + [2])
        else:
            article_ner_mask_ids = torch.LongTensor(article_ner_mask_dict["input_ids"])
        article_ner_mask_ids = article_ner_mask_ids.unsqueeze(0)
        
        
        caption_ids = self.tokenizer(caption, return_tensors="pt", truncation=True,  max_length=100)["input_ids"]
        if self.use_clip_tokenizer:
            import clip
            # caption_ids_clip = clip.tokenize(caption, context_length=100, truncate=True)
            caption_ids_clip = clip.tokenize(caption, truncate=True)
        else:
            caption_ids_clip = None
        # print(caption_ids)
        # ner_ids = [self.tokenizer(ner, return_tensors="pt")["input_ids"] for ner in new_ner_list]
        names_art_ids = self.tokenizer(concat_names_art, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        names_ids = self.tokenizer(concat_names, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]

        org_norp_art_ids = self.tokenizer(concat_org_norp_art, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        org_norp_ids = self.tokenizer(concat_org_norp, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]

        gpe_loc_art_ids = self.tokenizer(concat_gpe_loc_art, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        gpe_loc_ids = self.tokenizer(concat_gpe_loc, return_tensors="pt", truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        # name_ids = [self.tokenizer(name, return_tensors="pt")["input_ids"] for name in names]
        img_tensor = self.transform(img).unsqueeze(0)

        return {"article": article, "article_ids":article_ids,"article_ner_mask_ids":article_ner_mask_ids, "caption": caption, "caption_ids": caption_ids, "caption_ids_clip": caption_ids_clip, "names_art": new_names_art_list, "org_norp_art": new_org_norp_art_list, "gpe_loc_art": new_gpe_loc_art_list, "names_art_ids": names_art_ids, "org_norp_art_ids": org_norp_art_ids, "gpe_loc_art_ids": gpe_loc_art_ids, "names": names, "org_norp": org_norp, "gpe_loc": gpe_loc, "names_ids": names_ids, "org_norp_ids": org_norp_ids, "gpe_loc_ids": gpe_loc_ids, "all_gt_ner_ids":gt_ner_ids, "face_emb":face_emb, "obj_emb":obj_emb, "img_tensor":img_tensor}
    def __len__(self):
        return len(self.data_dict)





class GoodNewsDictDatasetEntityTypeFixLenEntPos(Dataset):
    def __init__(self, data_dict, data_base_dir, tokenizer, use_clip_tokenizer=False, entity_token_start="no", entity_token_end="no", transform=None, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50265):
        super().__init__()
        self.data_dict = copy.deepcopy(data_dict)
        self.face_dir = os.path.join(data_base_dir, "faces")
        self.obj_dir = os.path.join(data_base_dir, "objects")
        self.article_dir = os.path.join(data_base_dir, "articles_full")
        self.article_ner_mask_dir = os.path.join(data_base_dir, "articles_full_newsmep_ent_by_count")
        self.img_dir = os.path.join(data_base_dir, "images_processed")
        self.tokenizer = tokenizer
        self.use_clip_tokenizer = use_clip_tokenizer
        self.max_len = max_article_len
        self.transform = transform
        self.entity_token_start = entity_token_start
        self.entity_token_end = entity_token_end
        self.hash_ids = [*data_dict.keys()]
        self.max_ner_type_len = max_ner_type_len
        self.max_ner_type_len_gt = max_ner_type_len_gt

        self.retrieved_sent=retrieved_sent

        self.person_token_id = person_token_id
    
    def __getitem__(self, index):
        hash_id = self.hash_ids[index]
        img = Image.open(os.path.join(self.img_dir, f"{hash_id}.jpg")).convert('RGB')
        if self.data_dict[hash_id]["face_emb_dir"] != []:
            face_emb = np.load(os.path.join(self.face_dir, f"{hash_id}.npy"))
            names = self.data_dict[hash_id]["names"]
        else:
            face_emb = np.array([[]])
            names = []
        
        if self.data_dict[hash_id]["obj_emb_dir"] != []:
            obj_emb = np.load(os.path.join(self.obj_dir, f"{hash_id}.npy"))
        else:
            obj_emb = np.array([[]])
        
        if self.retrieved_sent:
            article = self.data_dict[hash_id]["sents_byclip"]
        else:
            with open(os.path.join(os.path.join(self.article_dir, f"{hash_id}.txt"))) as f:
                # article = f.readlines()
                article = f.read()
        
        # caption = self.data_dict[hash_id]["caption"]
        caption = unidecode.unidecode(self.data_dict[hash_id]["caption"])
        names = self.data_dict[hash_id]["names"]
        org_norp = self.data_dict[hash_id]["org_norp"]
        gpe_loc = self.data_dict[hash_id]["gpe_loc"]
        names_art = self.data_dict[hash_id]["names_art"]
        org_norp_art = self.data_dict[hash_id]["org_norp_art"]
        gpe_loc_art = self.data_dict[hash_id]["gpe_loc_art"]
        # article = preprocess_article(article)

        # delete repetitive names
        new_names_art_list = []
        for i in range(len(names_art)):
            if compare_ner(names_art[i], names_art[:i] + names_art[i+1:]):
                continue
            else:
                new_names_art_list.append(names_art[i])


        # delete repetitive orgs
        new_org_norp_art_list = []
        for i in range(len(org_norp_art)):
            if compare_ner(org_norp_art[i], org_norp_art[:i] + org_norp_art[i+1:]):
                continue
            else:
                new_org_norp_art_list.append(org_norp_art[i])
        # delete repetitive gpe
        new_gpe_loc_art_list = []
        for i in range(len(gpe_loc_art)):
            if compare_ner(gpe_loc_art[i], gpe_loc_art[:i] + gpe_loc_art[i+1:]):
                continue
            else:
                new_gpe_loc_art_list.append(gpe_loc_art[i])
        
        new_org_norp_gpe_loc_art_list = [*new_org_norp_art_list, *new_gpe_loc_art_list]
        org_norp_gpe_loc = [*org_norp, *gpe_loc]


        all_gt_ner = names + org_norp + gpe_loc
        # print(all_gt_ner)
        concat_gt_ner = concat_ner(all_gt_ner, self.entity_token_start, self.entity_token_end)
        # print(concat_gt_ner)
        gt_ner_ids = self.tokenizer(concat_gt_ner, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_ner_type_len_gt)["input_ids"]
        
        article_ids = self.tokenizer(article, return_tensors="pt", truncation=True, padding=True,max_length=self.max_len)["input_ids"]

        article_ner_mask_ids = torch.randn((1,512))


        with open(os.path.join(self.article_ner_mask_dir, f"{hash_id}.json")) as f:
            article_ner_mask_dict = json.load(f)

        person_id_positions = get_person_ids_position(article_ner_mask_dict["input_ids"], person_token_id=self.person_token_id, article_max_length=self.max_len)
        person_id_positions_cap = self.data_dict[hash_id]["name_pos_cap"]
        
        caption_ids = self.tokenizer(caption, return_tensors="pt", truncation=True,  max_length=100)["input_ids"]
        if self.use_clip_tokenizer:
            import clip
            # caption_ids_clip = clip.tokenize(caption, context_length=100, truncate=True)
            caption_ids_clip = clip.tokenize(caption, truncate=True)
        else:
            caption_ids_clip = None
            
        names_art_ids, _ = make_new_entity_ids(article, new_names_art_list, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)
        # concat_names_art = concat_ner(new_names_art_list, self.entity_token_start, self.entity_token_end)
        # names_art_ids = self.tokenizer(concat_names_art, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_ner_type_len)["input_ids"]
        

        names_ids_flatten, names_ids = make_new_entity_ids(caption, names, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)

        # names.append("<NONAME>")
        # names_ids = self.tokenizer(names, truncation=True, padding='max_length', max_length=self.max_ner_type_len_gt)["input_ids"]
        

        org_norp_gpe_loc_art_ids, _ = make_new_entity_ids(article, new_org_norp_gpe_loc_art_list, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)
        # concat_org_norp_gpe_loc_art = concat_ner(new_org_norp_gpe_loc_art_list, self.entity_token_start, self.entity_token_end)
        # org_norp_gpe_loc_art_ids = self.tokenizer(concat_org_norp_gpe_loc_art, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_ner_type_len)["input_ids"]
       

        org_norp_gpe_loc_ids_flatten, org_norp_gpe_loc_ids = make_new_entity_ids(caption, org_norp_gpe_loc, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)


        # org_norp_gpe_loc.append("NOORG")
        # org_norp_gpe_loc_ids = self.tokenizer(org_norp_gpe_loc, truncation=True, padding='max_length', max_length=self.max_ner_type_len_gt)["input_ids"]


        img_tensor = self.transform(img).unsqueeze(0)

        return {"article": article, "article_ids":article_ids, "article_ner_mask_ids":article_ner_mask_ids, "caption": caption, "caption_ids": caption_ids, "caption_ids_clip": caption_ids_clip, "names_art": new_names_art_list, "org_norp_gpe_loc_art": new_org_norp_gpe_loc_art_list, "names_art_ids": names_art_ids, "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids, "names": names, "org_norp_gpe_loc": org_norp_gpe_loc, "names_ids": names_ids, "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids, "all_gt_ner":all_gt_ner, "all_gt_ner_ids":gt_ner_ids, "face_emb":face_emb, "obj_emb":obj_emb, "img_tensor":img_tensor, "names_ids_flatten":names_ids_flatten, "org_norp_gpe_loc_ids_flatten":org_norp_gpe_loc_ids_flatten, "person_id_positions":person_id_positions, "person_id_positions_cap":person_id_positions_cap}
    def __len__(self):
        return len(self.data_dict)








def concat_ner(ner_list, entity_token_start, entity_token_end):
    concat_ner_list = []
    if entity_token_start == "no" or entity_token_end=="no":
        for ner in ner_list:
            concat_ner_list.extend(ner)
    elif entity_token_start == "|":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start])
            else:
                concat_ner_list.extend([ner])
    elif entity_token_start == "</s>":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start + entity_token_end])
            else:
                concat_ner_list.extend([ner])
    else:
        for ner in ner_list:
            # print(entity_token_start + " " + ner + " " + entity_token_end)
            concat_ner_list.extend([entity_token_start + " " + ner + " " + entity_token_end])
    # print(concat_ner_list)
    return " ".join(concat_ner_list)


def compare_ner(ner, ner_list):
    counter = 0
    for compare_ner in ner_list:
        if ner in compare_ner:
            counter += 1
        else:
            continue
    if counter > 0:
        return True
    else:
        return False

def preprocess_article(article):
    out = []
    for sent in article:
        if sent == "\n":
            continue
        else:
            out.append(sent.replace("\n", ""))
    return out



def get_entities(doc):
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "NORP", "GPE", "LOC"]:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                # 'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
                "position": [ent.start, ent.end],
            })
    return entities

def make_ner_dict_by_type(processed_doc, ent_list, ent_type_list):
    # make dict for unique ners with format as: {"Bush": PERSON_1}
    person_count = 1 # total count of PERSON type entities
    org_count = 1 # total count of ORG type entities
    gpe_count = 1 # total count of GPE type entities

    unique_ner_dict = {}
    new_ner_type_list = []

    for i, ent in enumerate(ent_list):
        if ent in unique_ner_dict.keys():
            new_ner_type_list.append(unique_ner_dict[ent])
        elif ent_type_list[i] == "PERSON":
            ner_type = "<PERSON>" + f"_{person_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            person_count += 1
        elif ent_type_list[i] == "ORG" or ent_type_list[i] == "NORP":
            ner_type = "<ORGNORP>" + f"_{org_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            org_count += 1
        elif ent_type_list[i] == "GPE" or ent_type_list[i] == "LOC":
            ner_type = "<GPELOC>" + f"_{gpe_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            gpe_count += 1
        
    entities_type = {} # dict with ner labels replaced by "PERSON_i", "ORG_j", "GPE_k"

    entities = get_entities(processed_doc)

    for i, ent in enumerate(entities):
        entities_type[i] = ent
        entities_type[i]["label"] = new_ner_type_list[i]
    # print(entities_type)

    start_pos_list = [sample["position"][0] for sample in entities_type.values()] # list of start positions for each entity
    # print(start_pos_list)
        
    return entities_type, start_pos_list, person_count, org_count, gpe_count


def make_new_articles_all_ent(processed_doc, start_pos_list, ent_length_list, entities_type_dict):
    # make new articles by replace PERSON/ORG/GPE type entities with their respective entity type
    counter = 0
    article_list = []
    article_list_unique_ner = []
    doc_len = len(processed_doc)
    ent_count = 0
    for i in range(doc_len):
        if i in start_pos_list and i+1 < doc_len:
            if processed_doc[i+1].is_punct or processed_doc[i+1].text == "'s":
                # if the entity is before punctuation or "'s"
                # we add it n-1 times concat with " ", 1 time with it self
                # n is the length of the tokenized entity from our tokenizer
                article_list_unique_ner.extend((ent_length_list[ent_count]-1) * [entities_type_dict[counter]["label"]+" "])
                article_list_unique_ner.append(entities_type_dict[counter]["label"])
                article_list.extend((ent_length_list[ent_count]-1) * [entities_type_dict[counter]["label"].split("_")[0]+ " "])
                article_list.append(entities_type_dict[counter]["label"].split("_")[0])
            else:
                article_list_unique_ner.extend(ent_length_list[ent_count] * [entities_type_dict[counter]["label"]+" "])
                article_list.extend(ent_length_list[ent_count] * [entities_type_dict[counter]["label"].split("_")[0]+" "])
            counter += 1 
            ent_count += 1
            start_pos_list = start_pos_list[1:]
        else:
            article_list_unique_ner.append(processed_doc[i].text_with_ws)
            article_list.append(processed_doc[i].text_with_ws)
    new_article_unique_ner = "".join(article_list_unique_ner)
    new_article = "".join(article_list)

    # print(new_article)
    # print(processed_doc)

    return new_article, new_article_unique_ner



def save_full_processed_articles_all_ent_by_count(data_dict, article_full_text_dir, article_all_ent_by_count_dir, article_all_ent_unique_by_count_dir, tokenizer):
    # goodnews_data: dataset of GoodNewsDatasetFullTxt class

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("merge_entities") # all recognized entities are counted as one word, we will count the length of the entities using the tokenizer of our choice

    for key in tqdm(data_dict.keys()):
        with open(os.path.join(article_full_text_dir, f"{key}.txt")) as f:
            article_full = f.read()
        
        processed_doc = nlp(article_full)
        entities = get_entities(processed_doc)
        
        ent_list = [ entities[i]["text"] for i in range(len(entities)) ]
        ent_type_list = [ entities[i]["label"] for i in range(len(entities)) ]
        
        entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(processed_doc, ent_list, ent_type_list)
        # print(entities_type)
        # ent_length_list = [len(tokenizer(ent)) for ent in ent_list]
        ent_length_list = [len(tokenizer(ent)["input_ids"])-2 for ent in ent_list]
        # print(ent_list)
        # print(ent_length_list)
        # new_article, new_article_unique_ner = make_new_articles_all_ent(processed_doc, start_pos_list, ent_length_list, entities_type)

        # article_all_ent_by_count_out_dir = os.path.join(article_all_ent_by_count_dir, f"{key}.txt")
        # article_all_ent_unique_by_count_out_dir = os.path.join(article_all_ent_unique_by_count_dir, f"{key}.txt")

        # with open(article_all_ent_by_count_out_dir, "w", encoding="utf-8") as f:
        #     f.write(new_article)
        # with open(article_all_ent_unique_by_count_out_dir, "w", encoding="utf-8") as f:
        #     f.write(new_article_unique_ner)
        

        article_ids_ner = make_new_article_ids_all_ent(article_full, ent_list, entities_type, tokenizer)
        # print(article_ids_ner)
        # input_ids_bart = torch.LongTensor(article_ids_ner)
        # input_ids_bart = input_ids_bart.unsqueeze(0)
        # print(input_ids_bart.size())
        # from transformers import BartModel
        # model = BartModel.from_pretrained("facebook/bart-base")
        # model.resize_token_embeddings(len(tokenizer))
        # print(model(input_ids=input_ids_bart))

        article_all_ent_by_count_out_dir = os.path.join(article_all_ent_by_count_dir, f"{key}.json")
        # article_all_ent_unique_by_count_out_dir = os.path.join(article_all_ent_unique_by_count_dir, f"{key}.txt")
        if not os.path.isfile(article_all_ent_by_count_out_dir):
            with open(article_all_ent_by_count_out_dir, "w", encoding="utf-8") as f:
                json.dump(article_ids_ner, f)
        # with open(article_all_ent_unique_by_count_out_dir, "w", encoding="utf-8") as f:
            # f.write(new_article_unique_ner)
        # print(len(tokenizer(article_full)["input_ids"]))
        # # print(tokenizer(article_full)["input_ids"])
        # print(len(article_ids_ner["input_ids"]))
        # print(article_full)


def make_new_article_ids_all_ent(article_full, ent_list, entities_type, tokenizer):
    article_ids_ner = tokenizer(article_full)["input_ids"]
    # article_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # in case entities were in the middle of the sentence
        # print(ent)
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # print(ent_ids_original)
        # if ent_ids_original in article_ids_ner:
        idx = find_first_sublist(article_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # print(ent_ids_original)
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            # print(tokenizer(ner_chain)["input_ids"][1:-1])
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # print(ent_ids_original)
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        # print(article_ids_ner)
        # # in case entities were in the middle of the sentence
        # ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # ner_chain_by_count = " ".join([entities_type[counter]["label"]] * len(ent_ids_original))
        # article_ids_ner_count = replace_sublist(article_ids_ner_count, ent_ids_original, tokenizer(ner_chain_by_count)["input_ids"][1:-1])
        # # in case entities were at the start of the sentence
        # ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
        # article_ids_ner_count = replace_sublist(article_ids_ner_count, ent_ids_original_start, tokenizer(ner_chain_by_count)["input_ids"][1:-1])
        
        counter += 1
    # print(len(article_ids_ner), len(tokenizer(article_full)["input_ids"]))
    # return article_ids_ner, article_ids_ner_count
    return {"input_ids":article_ids_ner}


def replace_sublist(seq, sublist, replacement):
    length = len(replacement)
    index = 0
    for start, end in iter(lambda: find_first_sublist(seq, sublist, index), None):
        seq[start:end] = replacement
        index = start + length
    return seq

def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index:index+length] == sublist:
            return index, index+length


def get_caption_with_ent_type(nlp, caption, tokenizer):
    processed_doc = nlp(caption)
    entities = get_entities(processed_doc)
        
    ent_list = [ entities[i]["text"] for i in range(len(entities)) ]
    ent_type_list = [ entities[i]["label"] for i in range(len(entities)) ]
        
    entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(processed_doc, ent_list, ent_type_list)

    new_caption, caption_ids_ner = make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer)
    return new_caption, caption_ids_ner


def make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer):
    caption_ids_ner = tokenizer(caption)["input_ids"]
    # caption_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # in case entities were in the middle of the sentence
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # if ent_ids_original in caption_ids_ner:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        counter += 1
    return tokenizer.decode(caption_ids_ner), caption_ids_ner


def add_name_pos_list_to_dict(data_dict, nlp, tokenizer):
    new_dict = {}
    for key, value in tqdm(data_dict.items()):
        new_dict[key] = {}
        new_dict[key] = value
        _, caption_ids_ner = get_caption_with_ent_type(nlp, value["caption"], tokenizer)
        position_list = get_person_ids_position(caption_ids_ner, person_token_id=50265, article_max_length=20, is_tgt_input=True)

        new_dict[key]["name_pos_cap"] = position_list
    return new_dict


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.add_special_tokens({"additional_special_tokens":['<PERSON>', "<ORGNORP>", "<GPELOC>"]})

    import spacy
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("merge_entities")


    with open("/DATADIR/GoodNews/test_dict_newsmep_ent.json") as f:
        test_dict_newsmep_ent = json.load(f)
    
    # save_full_processed_articles_all_ent_by_count(test_dict_newsmep_ent, "/DATADIR/GoodNews/goodnews/articles_full", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_by_count", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_unique_by_count", tokenizer)
    # # {"4fd299828eb7c8105d877504_0":{}}
    # # {"4fd292e98eb7c8105d86c97a_0":{}}


    test_dict_newsmep_ent_new =  add_name_pos_list_to_dict(test_dict_newsmep_ent, nlp, tokenizer)

    with open("/DATADIR/GoodNews/test_dict_newsmep_ent_cap_name_pos.json", "w") as f:
        json.dump(test_dict_newsmep_ent_new, f)


    with open("/DATADIR/GoodNews/val_dict_newsmep_ent.json") as f:
        val_dict_newsmep_ent = json.load(f)
    

    
    # # save_full_processed_articles_all_ent_by_count(val_dict_newsmep_ent, "/DATADIR/GoodNews/goodnews/articles_full", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_by_count", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_unique_by_count", tokenizer)

    val_dict_newsmep_ent_new =  add_name_pos_list_to_dict(val_dict_newsmep_ent, nlp, tokenizer)

    with open("/DATADIR/GoodNews/val_dict_newsmep_ent_cap_name_pos.json", "w") as f:
        json.dump(val_dict_newsmep_ent_new, f)


    with open("/DATADIR/GoodNews/train_dict_newsmep_ent.json") as f:
        train_dict_newsmep_ent = json.load(f)
    
    # # save_full_processed_articles_all_ent_by_count(train_dict_newsmep_ent, "/DATADIR/GoodNews/goodnews/articles_full", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_by_count", "/DATADIR/GoodNews/goodnews/articles_full_newsmep_ent_unique_by_count", tokenizer)


    train_dict_newsmep_ent_new =  add_name_pos_list_to_dict(train_dict_newsmep_ent, nlp, tokenizer)

    with open("/DATADIR/GoodNews/train_dict_newsmep_ent_cap_name_pos.json", "w") as f:
        json.dump(train_dict_newsmep_ent_new, f)
