import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=str, default="684331")
parser.add_argument("--gpu_ids", type=str, default="3")
parser.add_argument("--num_workers", type=int, default=16)

parser.add_argument("--plm_type", type=str, default="facebook/bart-base")
# parser.add_argument("--use_clip_feat", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--clip_type", type=str, default="ViT-B/32")
parser.add_argument("--ent_start_token", type=str, default="no")
parser.add_argument("--ent_end_token", type=str, default="no")

parser.add_argument("--enc_fusion_layer", nargs="+", type=int)
parser.add_argument("--dec_fusion_layer", nargs="+", type=int)
parser.add_argument("--use_img_trans", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--use_forget_gate", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--cross_attn_type", type=int, default=5)
parser.add_argument("--dim_common", type=int, default=768)
parser.add_argument("--n_attn_heads", type=int, default=12)

parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--max_length", type=int, default=100)

parser.add_argument("--data_type", type=str, default= "nytimes")
parser.add_argument("--data_dir", type=str, default= "DATADIR")

parser.add_argument("--prompt_size", type=int, default=8)
parser.add_argument("--model_name", type=str, default="MODELNAME")
parser.add_argument("--model_dir", type=str, default="MODELDIR")

parser.add_argument("--num_sentences", type=int, default=8)

parser.add_argument("--dict_type", type=str, default="tune")

parser.add_argument("--length_penalty", type=float, default=1)

args = parser.parse_args()


def shift_tokens_right(input_ids, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def create_src_mask_bart(input_ids):
    src_padding_mask = (input_ids == 1)
    src_mask = (src_padding_mask<1)
    src_mask = src_mask.int().type(torch.int64)
    src_mask = src_mask.to(input_ids.device)
    return src_mask


# def extract_clip_img_feat(clip_model, x):
#     vit_backbone = clip_model.eval().visual
#     dtype = vit_backbone.conv1.weight.dtype
#     DEVICE = x.device
#     x = vit_backbone.conv1(x.type(dtype))
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#     x = torch.cat([vit_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=DEVICE), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#     x = x + vit_backbone.positional_embedding.to(x.dtype)
#     x = vit_backbone.ln_pre(x)

#     x = x.permute(1, 0, 2)  # NLD -> LND
#     x = vit_backbone.transformer(x)
#     x = x.permute(1, 0, 2)  # LND -> NLD
#     # x = self.vit_backbone.ln_post(x[:, 0, :]) # [1, 50, 768] --> [1, 768] ([CLS])
#     x = vit_backbone.ln_post(x[:, 1: ,:])
#     x = x.float()

#     return x



def extract_clip_img_feat(clip_model, x):
    with torch.no_grad():
        vit_backbone = clip_model.eval().visual
        dtype = vit_backbone.conv1.weight.dtype
        DEVICE = x.device
        x = vit_backbone.conv1(x.type(dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=DEVICE), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit_backbone.positional_embedding.to(x.dtype)
        x = vit_backbone.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = vit_backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.vit_backbone.ln_post(x[:, 0, :]) # [1, 50, 768] --> [1, 768] ([CLS])
        x_cls = vit_backbone.ln_post(x[:, 0, :])
        x_cls = x_cls.float()
        x = vit_backbone.ln_post(x[:, 1: ,:])
        x = x.float()
    return x, x_cls


def extract_clip_img_feat_ner(clip_model, x):
    vit_backbone = clip_model.eval().visual
    dtype = vit_backbone.conv1.weight.dtype
    DEVICE = x.device
    x = vit_backbone.conv1(x.type(dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([vit_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=DEVICE), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + vit_backbone.positional_embedding.to(x.dtype)
    x = vit_backbone.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = vit_backbone.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    # x = self.vit_backbone.ln_post(x[:, 0, :]) # [1, 50, 768] --> [1, 768] ([CLS])
    x_cls = vit_backbone.ln_post(x[:, 0, :])
    x_cls = x_cls.float()
    x = vit_backbone.ln_post(x[:, 1: ,:])
    x = x.float()

    return x, x_cls


def gen_caption_from_loader_bart(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        # src_ids, tgt_sent = batch["article_ids"], batch["caption"]    
        # src_mask = create_src_mask_bart(src_ids)
        src_ids, tgt_sent, img_tensor = batch["article_ids"], batch["caption"], batch["img_tensor"]
        src_ids = src_ids.to(DEVICE)
        img_tensor = img_tensor.to(DEVICE)
        
        img_feat, _ = extract_clip_img_feat(model.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
            
        gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, length_penalty=args.length_penalty)
        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def gen_caption_from_loader_bart_prompt(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        # src_ids, tgt_sent = batch["article_ids"], batch["caption"]    
        # src_mask = create_src_mask_bart(src_ids)
        src_ids, tgt_sent, img_tensor = batch["article_ids"], batch["caption"], batch["img_tensor"]
        src_ids = src_ids.to(DEVICE)
        img_tensor = img_tensor.to(DEVICE)
        
        img_feat, _ = extract_clip_img_feat(model.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)
            
        gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, length_penalty=args.length_penalty)
        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score




def gen_caption_from_loader_bart_prompt_ent(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    if "," in args.gpu_ids:
        model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        # src_ids, tgt_sent = batch["article_ids"], batch["caption"]    
        # src_mask = create_src_mask_bart(src_ids)
        src_ids, tgt_sent, ner_ids, name_ids, img_tensor = batch["article_ids"],  batch["caption"], batch["ner_ids"], batch["name_ids"], batch["img_tensor"]

        # src_ids, tgt_sent, img_tensor = batch["article_ids"], batch["caption"], batch["img_tensor"]
        src_ids = src_ids.to(DEVICE)
        img_tensor = img_tensor.to(DEVICE)

        # if args.only_name:
            # ner_ids = name_ids.to(DEVICE)
        # else:
            # ner_ids = ner_ids.to(DEVICE)
        if "," in args.gpu_ids:
            img_feat,_ = extract_clip_img_feat_ner(model.module.clip_model, img_tensor)
        else:
            img_feat,_ = extract_clip_img_feat_ner(model.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)
        ner_mask = create_src_mask_bart(ner_ids)
        if "," in args.gpu_ids:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, ner_ids=ner_ids, ner_mask=ner_mask, length_penalty=args.length_penalty)
        else:
            gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, ner_ids=ner_ids, ner_mask=ner_mask,length_penalty=args.length_penalty)
            
        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)
        
        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def gen_caption_from_loader_bart_prompt_ent_type(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        src_ids, tgt_ids, tgt_ids_clip, img_tensor, names_art_ids, org_art_ids, gpe_art_ids, tgt_sent = batch["article_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"], batch["names_art_ids"], batch["org_art_ids"], batch["gpe_art_ids"], batch["caption"],

        src_ids = src_ids.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)
        org_art_ids = org_art_ids.to(DEVICE)
        gpe_art_ids = gpe_art_ids.to(DEVICE)

        img_tensor = img_tensor.to(DEVICE)
        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)
        
        names_art_mask = create_src_mask_bart(names_art_ids)
        org_art_mask = create_src_mask_bart(org_art_ids)
        gpe_art_mask = create_src_mask_bart(gpe_art_ids)

        # print(names_art_mask.size(), org_art_mask.size(), gpe_art_mask.size())
        ner_art_mask = create_src_mask_bart(torch.zeros(names_art_ids.size()))
        # print(ner_art_mask.size())

        gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, name_ids=names_art_ids, org_ids=org_art_ids, gpe_ids=gpe_art_ids, name_mask=names_art_mask, org_mask=org_art_mask, gpe_mask=gpe_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True, length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score




def gen_caption_from_loader_bart_prompt_ent_type_newsmep(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        src_ids, tgt_ids, tgt_ids_clip, img_tensor, names_art_ids, org_norp_art_ids, gpe_loc_art_ids, tgt_sent = batch["article_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"], batch["names_art_ids"], batch["org_norp_art_ids"], batch["gpe_loc_art_ids"], batch["caption"],

        src_ids = src_ids.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)
        org_norp_art_ids = org_norp_art_ids.to(DEVICE)
        gpe_loc_art_ids = gpe_loc_art_ids.to(DEVICE)

        img_tensor = img_tensor.to(DEVICE)
        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)
        
        names_art_mask = create_src_mask_bart(names_art_ids)
        org_norp_art_mask = create_src_mask_bart(org_norp_art_ids)
        gpe_loc_art_mask = create_src_mask_bart(gpe_loc_art_ids)

        # print(names_art_mask.size(), org_art_mask.size(), gpe_art_mask.size())
        ner_art_mask = torch.ones(names_art_ids.size())
        # print(ner_art_mask.size())

        if "clipcap" in args.model_name:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, image_len=None, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True,length_penalty=args.length_penalty)
        else:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True,length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        print(f"gt:{gt_unidecode}")
        print(f"gen:{gen_unidecode}")

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score




def gen_caption_from_loader_bart_noent(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    #     model = model.module
    model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        src_ids, src_ner_mask_ids, tgt_ids, tgt_ids_clip, img_tensor, tgt_sent = batch["article_ids"], batch["article_ner_mask_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"], batch["caption"],

        src_ids = src_ids.to(DEVICE)
        src_ner_mask_ids = src_ner_mask_ids.to(DEVICE)

        img_tensor = img_tensor.to(DEVICE)

        # if "," in args.gpu_ids:
        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensor)
        # else:
        #     img_feat,img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)
        

        # if "," in args.gpu_ids:
        if "clipcap" in args.model_name:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, image_len=None, length_penalty=args.length_penalty)
        else:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, length_penalty=args.length_penalty)
        # else:
        #     if args.prompt_mlp_type == "clipcap":
        #         gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, image_len=None,)
        #     else:
        #         gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None,)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def gen_caption_from_loader_bart_face_obj(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    # model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}

        src_ids, tgt_ids, img_tensors, names_art_ids, org_norp_gpe_loc_art_ids, face_emb, obj_emb, tgt_sent = batch["article_ids"],  batch["caption_ids"], batch["img_tensor"], batch["names_art_ids"], batch["org_norp_gpe_loc_art_ids"], batch["face_emb"], batch["obj_emb"], batch["caption"]

        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)
        obj_emb = obj_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)
        org_norp_gpe_loc_art_ids = org_norp_gpe_loc_art_ids.to(DEVICE)


        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)

        
        src_mask = create_src_mask_bart(src_ids)

        names_art_mask = create_src_mask_bart(names_art_ids)
        org_norp_gpe_loc_art_mask = create_src_mask_bart(org_norp_gpe_loc_art_ids)
        # ner_art_mask = torch.ones(names_art_mask.size())
        ner_art_mask = torch.ones((1,20))


        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        src_mask = torch.cat((src_mask, visual_mask), dim=1)

        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        obj_mask = create_src_mask_bart(obj_emb[:, :, -1])


        if "clipcap" in args.model_name:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, face_features=face_emb, face_mask=face_mask, obj_features=obj_emb, obj_mask=obj_mask, name_ids=names_art_ids, org_gpe_ids=org_norp_gpe_loc_art_ids, name_mask=names_art_mask, org_gpe_mask=org_norp_gpe_loc_art_mask, ner_mask=ner_art_mask, add_cap_ner_attn=True, length_penalty=args.length_penalty)
        else:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, face_features=face_emb, face_mask=face_mask, obj_features=obj_emb, obj_mask=obj_mask, name_ids=names_art_ids, org_gpe_ids=org_norp_gpe_loc_art_ids, name_mask=names_art_mask, org_gpe_mask=org_norp_gpe_loc_art_mask, ner_mask=ner_art_mask, add_cap_ner_attn=True, length_penalty=args.length_penalty)
        
        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)

        print(f"gt:{gt_unidecode}")
        print(f"gen:{gen_unidecode}")

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score





def gen_caption_from_loader_bart_prompt_ent_type_newsmep_fixlen(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        src_ids, tgt_ids, tgt_ids_clip, img_tensor, names_art_ids, org_norp_art_ids, gpe_loc_art_ids, tgt_sent = batch["article_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"], batch["names_art_ids"], batch["org_norp_art_ids"], batch["gpe_loc_art_ids"], batch["caption"],

        src_ids = src_ids.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)
        org_norp_art_ids = org_norp_art_ids.to(DEVICE)
        gpe_loc_art_ids = gpe_loc_art_ids.to(DEVICE)

        img_tensor = img_tensor.to(DEVICE)
        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensor)

        src_mask = create_src_mask_bart(src_ids)
        visual_mask = torch.ones((src_mask.size(0), args.prompt_size)).to(src_mask.device)
        if "enc-self-dec-cross" not in args.model_name:
            src_mask = torch.cat((src_mask, visual_mask), dim=1)
        
        names_art_mask = create_src_mask_bart(names_art_ids)
        org_norp_art_mask = create_src_mask_bart(org_norp_art_ids)
        gpe_loc_art_mask = create_src_mask_bart(gpe_loc_art_ids)

        # print(names_art_mask.size(), org_art_mask.size(), gpe_art_mask.size())
        names_ids = batch["names_ids"]
        ner_art_mask = torch.ones(names_ids.size())
        # print(ner_art_mask.size())

        if "clipcap" in args.model_name:
            if "enc-self-dec-cross" in args.model_name:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True, length_penalty=args.length_penalty)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, image_len=None, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True, length_penalty=args.length_penalty)
        else:
            if "enc-self-dec-cross" in args.model_name:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True, length_penalty=args.length_penalty)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, image_len=None, name_ids=names_art_ids, org_ids=org_norp_art_ids, gpe_ids=gpe_loc_art_ids, name_mask=names_art_mask, org_mask=org_norp_art_mask, gpe_mask=gpe_loc_art_mask, ner_mask=ner_art_mask, add_img_ner_attn=True, length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(gen_cap)

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        # print(f"gt:{gt_unidecode}")
        # print(f"gen:{gen_unidecode}")

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = tgt_sent[0]
        out_dict[step]["gen"] = gen_cap
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score


def gen_caption_from_loader_bart_facenameID(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    # if "bart-large" in model_name:
    #     model = model
    # else:
    #     model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}

        src_ids, tgt_sent, img_tensors, face_emb, names_art_ids, = batch["article_ids"], batch["caption"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"],
        src_ids = src_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)

        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)


        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)
        # img_feat,img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensors)


        src_mask = create_src_mask_bart(src_ids)
        
        # print(names_mask.size(), org_norp_mask.size(), gpe_loc_mask.size())
        # ner_mask = create_src_mask_bart(torch.zeros(names_art_ids.size()))
        ner_mask = torch.ones((args.test_batch_size, 20))
        # print(ner_mask.size())
        ner_mask = ner_mask.to(DEVICE)


        if "clipcap" in args.model_name:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)
            # gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)

        else:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        # print(gen_unidecode)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def gen_caption_from_loader_bart_onlyvis(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}

        src_ids, tgt_sent, img_tensors, = batch["article_ids"], batch["caption"], batch["img_tensor"]
        src_ids = src_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)

        src_mask = create_src_mask_bart(src_ids)

        img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)

        src_mask = create_src_mask_bart(src_ids)


        if "clipcap" in args.model_name:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls, length_penalty=args.length_penalty)
        else:
            gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat, length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        # print(gen_unidecode)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score






def gen_caption_from_loader_bart_match_onlyvis(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    # if "," in args.gpu_ids:
    model = model.module
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}

        src_ids, tgt_sent, img_tensors, face_emb, names_art_ids, = batch["article_ids"], batch["caption"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"],
        src_ids = src_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)

        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)


        if "," in args.gpu_ids:
            img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat,img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensors)

        src_mask = create_src_mask_bart(src_ids)
        
        # print(names_mask.size(), org_norp_mask.size(), gpe_loc_mask.size())
        # ner_mask = create_src_mask_bart(torch.zeros(names_art_ids.size()))
        ner_mask = torch.ones((args.test_batch_size, 20))
        # print(ner_mask.size())
        ner_mask = ner_mask.to(DEVICE)

        if "," in args.gpu_ids:
            if "clipcap" in model_name:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)
        else:
            if "clipcap" in model_name:
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)
            else:
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True, length_penalty=args.length_penalty)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        # print(gen_unidecode)

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()



if __name__ == "__main__":
    import os
    from src.utils.misc import set_random_seed, get_logger
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # set_random_seed(int(args.seed))
    import torch
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.multiprocessing.set_sharing_strategy('file_system')
    import clip
    from src.data.goodnews_dataset_entity import GoodNewsDictDatasetEntity, pad_sequence, collate_fn_goodnews_entity, GoodNewsDictDatasetEntityParaCLIP, collate_fn_goodnews_entity_para_clip

    from venv import create
    
    from numpy import iterable
    
    import json
    from tqdm import tqdm
    
    from src.data.nytimes_dataset_newsmap_ent_face_obj import NYTimesDictDatasetEntityType, collate_fn_nytimes_entity_type
    # from src.data.goodnews_dataset_entity import GoodNewsDictDatasetEntity, pad_sequence, collate_fn_goodnews_entity
    from torch.utils.data import DataLoader
    from src.utils.generation_utils import beam_search, greedy_search, get_prob
    from transformers import get_linear_schedule_with_warmup, BartTokenizer, PreTrainedTokenizerFast
    from src.models.modeling_bart_vg_gplm_full_clip_inside import BartForMultiModalGeneration
    from torchvision import models, transforms
    import torch.optim as optim
    # from torchtext.data import get_tokenizer
    # from torchtext.data.metrics import bleu_score
    # import clip
    # from src.utils.clip_features import ClipViTFeat
    import re
    import types
    import numpy as np
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    import unidecode


    # tokenizer = BartTokenizer.from_pretrained(args.plm_type)
    if args.plm_type.startswith("ainize"):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    else:
        tokenizer = BartTokenizer.from_pretrained(args.plm_type)
    
    
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # model = BartForMultiModalGeneration.from_pretrained(args.plm_type, enc_fusion_layer=args.enc_fusion_layer, dec_fusion_layer=args.dec_fusion_layer, use_img_trans=args.use_img_trans, use_forget_gate=args.use_forget_gate, cross_attn_type=args.cross_attn_type, dim_common=args.dim_common, n_attn_heads=args.n_attn_heads, img_size=768, clip_model=clip_model)

    img_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    
    local_rank = int(os.environ['LOCAL_RANK'])
    # local_rank = args.local_rank
    # print(local_rank)
    # torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    DEVICE = torch.device("cuda", local_rank)

    model_name = args.model_name
    # model = torch.load(os.path.join("/cw/working-gimli/tingyu/news_cap/new", model_name+".pt"), map_location="cpu")
    model = torch.load(os.path.join(args.model_dir, model_name+".pt"), map_location="cpu")
    model = model.to(DEVICE)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=[local_rank])
   
    # print(model)

    clip_model, clip_preprocess = clip.load(args.clip_type, device=DEVICE)
    # clip_model = torch.load("/cw/working-eowyn/tingyu/news_cap/goodnewsCLIPViT-B-16_dict-full_seed684331_bsz128_lr5e-06-0.1_16epoch_warmup0.05last.pt")

    num_paragraph = 8
    # gen_caption_from_loader_bart_facenameID

    if "facenameID" in model_name and "retrieve" in model_name or "self-light" in model_name or "FNID" in model_name and "retri" in model_name:
        tokenizer_dataset = BartTokenizer.from_pretrained(args.plm_type)
        tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent_ent_pos import GoodNewsDictDatasetEntityTypeFixLenEntPos, collate_fn_goodnews_entity_type
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")    

            if args.dict_type == "notune":
                print("no tuned CLIP for retrieval")
                with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_clip{args.num_sentences}sent_name_pos_notune.json")) as f:
                    test_dict = json.load(f)
            else:
                print("tuned CLIP for retrieval")
                with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                    test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)

        else:
            from src.data.nytimes_dataset_newsmap_ent_article_seg_ent_pos import NYTimesDictDatasetEntityTypeFixLenEntPos, collate_fn_nytimes_entity_type
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")

            if args.dict_type == "notune":
                print("no tuned CLIP for retrieval")
                with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_clip{args.num_sentences}sent_name_pos_notune.json")) as f:
                    test_dict = json.load(f)
            else:
                print("tuned CLIP for retrieval")
                with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                    test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
    elif "FNID" in model_name and "retri" not in model_name:
        tokenizer_dataset = BartTokenizer.from_pretrained(args.plm_type)
        tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent_ent_pos import GoodNewsDictDatasetEntityTypeFixLenEntPos, collate_fn_goodnews_entity_type
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")    

            with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)

        else:
            from src.data.nytimes_dataset_newsmap_ent_article_seg_ent_pos import NYTimesDictDatasetEntityTypeFixLenEntPos, collate_fn_nytimes_entity_type
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            
            print("no retrieved sentences")
            
            with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
    elif "onlyvis" in model_name:
        tokenizer_dataset = BartTokenizer.from_pretrained(args.plm_type)
        tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent_ent_pos import GoodNewsDictDatasetEntityTypeFixLenEntPos, collate_fn_goodnews_entity_type
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")    

            print("no retrieved sentences")

            with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)

        else:
            from src.data.nytimes_dataset_newsmap_ent_article_seg_ent_pos import NYTimesDictDatasetEntityTypeFixLenEntPos, collate_fn_nytimes_entity_type
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")

            print("no retrieved sentences")

            with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start="<ENT>", entity_token_end="<ENT>", transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
    elif "fix" in model_name and "Face" in model_name:
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent_face_obj import GoodNewsDictDatasetEntityTypeFixLen, collate_fn_goodnews_entity_type_fix_len
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
            with open("/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict_newsmep_ent.json") as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLen(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=120, max_ner_type_len_gt=20)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type_fix_len)
        else:
            from src.data.nytimes_dataset_newsmap_ent_face_obj import NYTimesDictDatasetEntityTypeFixLen, collate_fn_nytimes_entity_type_fix_len
            with open(os.path.join("/cw/liir_data/NoCsBack/NewsCap/NYTimes/test_dict_newsmep_ent_cleaned.json")) as f:
                test_dict = json.load(f)
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            test_data = NYTimesDictDatasetEntityTypeFixLen(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=120, max_ner_type_len_gt=20)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type_fix_len)
    elif "fix" in model_name:
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent import GoodNewsDictDatasetEntityTypeFixLen, collate_fn_goodnews_entity_type
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
            with open("/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict_newsmep_ent.json") as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLen(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)
        else:
            from src.data.nytimes_dataset_newsmap_ent import NYTimesDictDatasetEntityTypeFixLen, collate_fn_nytimes_entity_type
            with open(os.path.join("/cw/liir_data/NoCsBack/NewsCap/NYTimes/test_dict_newsmep_ent_cleaned.json")) as f:
                test_dict = json.load(f)
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            test_data = NYTimesDictDatasetEntityTypeFixLen(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
    elif "newsmep" in model_name or "NERnone" in model_name:
        if args.data_type == "goodnews":
            from src.data.goodnews_dataset_entity_type_newsmep_ent import GoodNewsDictDatasetEntityType, collate_fn_goodnews_entity_type
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
            with open("/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict_newsmep_ent.json") as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityType(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=128)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)
        else:
            with open(os.path.join("/cw/liir_data/NoCsBack/NewsCap/NYTimes/test_dict_newsmep_ent_cleaned.json")) as f:
                test_dict = json.load(f)
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            test_data = NYTimesDictDatasetEntityType(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=512, max_ner_type_len=128)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
    elif "NERtype" in model_name:
        from src.data.goodnews_dataset_entity_type import GoodNewsDictDatasetEntityType, collate_fn_goodnews_entity_type
        data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")

        with open("/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict_ner.json") as f:
            test_dict = json.load(f)
        test_data = GoodNewsDictDatasetEntityType(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform)
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)
    
    elif "para" not in model_name:
        data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")

        with open("/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict.json") as f:
            test_dict = json.load(f)
        test_data = GoodNewsDictDatasetEntity(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform)
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity)

    else:
        data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
        with open(f"/cw/liir_data/NoCsBack/NewsCap/GoodNews/test_dict_trained_clip_{num_paragraph}para.json") as f:
            test_dict = json.load(f)
        test_data = GoodNewsDictDatasetEntityParaCLIP(test_dict, data_base_dir, tokenizer, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, img_dir=args.img_dir)
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_para_clip)

    
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []
    

    # if "facenameID" in model_name or "self-light" in model_name or "FNID" in model_name and "retri" in model_name:
    if "Match" in model_name and "onlyvis" in model_name:
        print("decode for onlyvis, match")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_match_onlyvis(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "facenameID" in model_name or "self-light" in model_name or "FNID" in model_name:
        print("decode using prompt-based model with SECLA")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_facenameID(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "onlyvis" in model_name:
        print("decode using model with SECLA, onlyvis")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_onlyvis(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "fixlen" in model_name:
        print("decode using prompt-based model with newsmep entities and fix length entity input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_prompt_ent_type_newsmep_fixlen(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "Face" in model_name:
        print("decode using prompt-based model with face and object input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_face_obj(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "NERnone" in model_name:
        print("decode using prompt-based model without entity input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_noent(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "newsmep" in model_name:
        print("decode using prompt-based model with entity type (NewsMEP) input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_prompt_ent_type_newsmep(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "NERtype" in model_name:
        print("decode using prompt-based model with entity type input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_prompt_ent_type(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "prompt" in model_name and "_ent" in model_name:
        print("decode using prompt-based model with entity input")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_prompt_ent(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    elif "prompt" in model_name:
        print("decode using prompt-based model")
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart_prompt(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    else:
        test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    

    print(model_name)
    print({"blue1":blue1, "blue2":blue2, "blue3":blue3, "blue4":blue4, "rouge_score":rouge_score, "meteor_score":meteor_score, "cider_score":cider_score})
    
    with open(os.path.join("OUTPUTDIR", model_name+f"beam{args.beam_size}_max{args.max_length}t_S{args.seed}_lp{args.length_penalty}.json"), "w") as f:
        json.dump(test_out_dict, f)
    
    print(f"beam{args.beam_size}_max{args.max_length}t_S{args.seed}_lp{args.length_penalty}")

    

    
    

   
