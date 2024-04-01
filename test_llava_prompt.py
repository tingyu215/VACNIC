from transformers import LlavaForConditionalGeneration, LlavaProcessor, LlavaConfig
from transformers import BitsAndBytesConfig
import torch
from tqdm import tqdm
import unidecode
import re
import numpy as np
from PIL import Image
import requests
import os
import json
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)

parser.add_argument("--use_opt", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--use_retrieval", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--data_type", type=str, default="goodnews")

args = parser.parse_args()

def gen_caption_from_loader_llava(model, data_loader, processor, beam_size, max_length, use_opt=True):
    out_dict = {}
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        raw_article, tgt_sent, raw_img= batch["article"], batch["caption"], batch["raw_img"]
        # do not use any transform in dataloader.
        if args.use_retrieval:
            prompts = []
            for i in range(len(raw_img)):
                prompts.append(f"USER: <image>\nNews article:{raw_article[i]}Generate news image caption:\nASSISTANT:")
        else:
            prompts = ["USER: <image>\nGenerate news image caption:\nASSISTANT:"] * len(raw_img)
        inputs = processor(prompts, images=raw_img, padding=True, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)
        gen_cap = processor.batch_decode(output, skip_special_tokens=True)
        gen_cap = [cap.split("ASSISTANT:")[-1] for cap in gen_cap]

        out_dict[step]["gt"] = tgt_sent
        out_dict[step]["gen"] = gen_cap

        
    
    return out_dict


if __name__ == "__main__":
    from src.data.dataset_entity_type_newsmep_blip import GoodNewsDictDatasetEntityTypeFixLenEntPos, collate_fn_goodnews_entity_type, NYTimesDictDatasetEntityTypeFixLenEntPos, collate_fn_nytimes_entity_type
    from transformers import BartTokenizer
    from torchvision import transforms

    tokenizer_dataset = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})

    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    data_dir = "/DATADIR/NewsCap"
    
    
    if args.use_retrieval:
        if args.data_type == "goodnews":
            data_base_dir = os.path.join(data_dir, "GoodNews/goodnews")
            with open(os.path.join(data_dir, f"GoodNews/test_dict_newsmep_ent_clip8sent_contras_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_goodnews_entity_type)
        else:
            data_base_dir = os.path.join(data_dir, "NYTimes/nytimes")

            with open(os.path.join(data_dir, f"NYTimes/test_dict_newsmep_ent_seg_clip10sent_contras_name_pos_new.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_nytimes_entity_type)

    else:
        if args.data_type == "goodnews":
            data_base_dir = os.path.join(data_dir, "GoodNews/goodnews")
            with open(os.path.join(data_dir, "GoodNews/test_dict_newsmep_ent_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_goodnews_entity_type)
        else:
            data_base_dir = os.path.join(data_dir, "NYTimes/nytimes")
            with open(os.path.join(data_dir, f"NYTimes/test_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_nytimes_entity_type)


    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", quantization_config=quantization_config)
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",  device_map="auto")

    output_dict = gen_caption_from_loader_llava(model, test_loader, processor, beam_size=5, max_length=50, use_opt=args.use_opt)

    with open(f"/OUTDIR/llava_{args.data_type}_retrieve{args.use_retrieval}_prompt.json", "w") as f:
        json.dump(output_dict, f)


