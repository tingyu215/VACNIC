
import argparse
from cmath import nan

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, default="684331")
parser.add_argument("--gpu_ids", type=str, default="1")
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--article_max_length", type=int, default=512)
parser.add_argument("--caption_max_length", type=int, default=100)
parser.add_argument("--plm_type", type=str, default="facebook/bart-base")
parser.add_argument("--clip_type", type=str, default="ViT-B/32")
parser.add_argument("--ent_start_token", type=str, default="no")
parser.add_argument("--ent_end_token", type=str, default="no")

parser.add_argument("--enc_fusion_layer", nargs="+", type=int)
parser.add_argument("--dim_common", type=int, default=768)

parser.add_argument("--warmup_rate", type=float, default=0.05)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=1)
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--lr_bart", type=float, default = 1e-4)
parser.add_argument("--lr_clip", type=float, default = 5e-6)
parser.add_argument("--weight_decay", type=float, default = 1e-5)
parser.add_argument("--clip_norm", type=float, default = 0.1)


parser.add_argument("--data_type", type=str, default= "nytimes")
parser.add_argument("--data_dir", type=str, default= ".")
parser.add_argument("--out_dir", type=str, default= ".")

parser.add_argument("--mapping_loss_type", type=str, default= "contrastive")

parser.add_argument("--trained_clip", type=str, default="no")
parser.add_argument("--clip_dir", type=str, default=".")
parser.add_argument("--no_clip_loss", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--prompt_size", type=int, default=10)
parser.add_argument("--use_vis_cls", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--max_ner_type_len", type=int, default=80)
parser.add_argument("--max_ner_type_len_gt", type=int, default=20)

parser.add_argument("--freeze_clip", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--prompt_mlp_type", type=str, default="clipcap")
parser.add_argument("--map_size", nargs="+", type=int)

parser.add_argument("--no_mapping", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--mapping_loss_weight", type=float, default = 1.0)
parser.add_argument("--img_size", type=int, default=768)

parser.add_argument("--only_image", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--use_secla", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--num_sentences", type=int, default=8)

parser.add_argument("--adapter_dim", type=int, default=96)
parser.add_argument("--project_name", type=str, default="news_cap")
parser.add_argument("--experiment_name", type=str, default="t5_retrieval_goodnews")

parser.add_argument("--offline_wandb", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--perturb", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--no_clip_norm", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--init_attn_weight", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--do_retrieval", action="store_true")


args = parser.parse_args()


def prep_for_training(model, train_size, DEVICE):
    if "," in args.gpu_ids:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    model.to(DEVICE)
    if "," in args.gpu_ids:
        optimizer_bart = optim.AdamW(list(model.module.model.parameters()) + list(model.module.lm_head.parameters()),betas= (0.9, 0.999), lr=args.lr_bart, eps=1e-8, weight_decay=args.weight_decay)

    else:
        optimizer_bart = optim.AdamW(list(model.model.parameters()) + list(model.lm_head.parameters()),betas= (0.9, 0.999), lr=args.lr_bart, eps=1e-8, weight_decay=args.weight_decay)

    num_training_steps = args.num_epoch * train_size / args.train_batch_size
    num_warmup_steps = args.warmup_rate * num_training_steps
    
    scheduler_bart = get_linear_schedule_with_warmup(optimizer_bart,
                                                num_warmup_steps,
                                                num_training_steps)

    return model, optimizer_bart, scheduler_bart



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

def train_epoch(model, loss_fn, train_dataloader, optimizer_bart, scheduler_bart, epoch, DEVICE):
    model.train()
    tr_loss = 0
    tr_txt_loss = 0
    nb_tr_steps = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        src_ids, tgt_ids, tgt_ids_clip, img_tensors = batch["article_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"]
        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)

        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        src_mask = create_src_mask_bart(src_ids)

        if "," in args.gpu_ids:
            img_feat, img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat, img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensors)

        if args.prompt_mlp_type == "clipcap":
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat_cls)
        else:
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat)
        logits = output["logits"]

        
        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
        tr_txt_loss += txt_loss.item()
        
        loss = txt_loss
        loss.backward()
        if not args.no_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        tr_loss += loss.item()
        nb_tr_steps += 1

        if loss == nan:
            print(batch)
        
        optimizer_bart.step()
        scheduler_bart.step()
        optimizer_bart.zero_grad()
        

        wandb.log({"loss": loss})
        wandb.log({"text loss": txt_loss})

    return tr_loss / nb_tr_steps


def eval_epoch(model, loss_fn, val_dataloader, DEVICE):
    model.eval()
    val_loss = 0
    nb_val_steps = 0
    out_dict = {}
    for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        out_dict[step] = {}
        
        src_ids, tgt_ids, tgt_sent, tgt_ids_clip, img_tensors, face_emb, names_art_ids, = batch["article_ids"], batch["caption_ids"], batch["caption"], batch["caption_ids_clip"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"],
        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        

        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        src_mask = create_src_mask_bart(src_ids)


        if "," in args.gpu_ids:
            img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat,img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensors)


        src_mask = create_src_mask_bart(src_ids)

        if args.prompt_mlp_type == "clipcap":
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat_cls)
        else:
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat)
        logits = output["logits"]

        out_dict[step]["logit_output"] = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.argmax(logits[i], dim=-1))) for i in range(logits.shape[0])]

        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))

        loss = txt_loss
        out_dict[step]["gt_cap"] = tgt_sent

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        else:
            loss = loss
        val_loss += loss.item()
        nb_val_steps += 1

        wandb.log({"validation loss": loss})

    return val_loss / nb_val_steps, out_dict


def train(model, loss_fn, train_dataloader, val_dataloader, test_dataloader, optimizer_bart, scheduler_bart, model_name, DEVICE):
    train_losses = []
    val_losses = []

    min_val_loss = 999
    wandb.watch(model)
    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer_bart, scheduler_bart, epoch_i, DEVICE)

        val_loss, out_dict = eval_epoch(model, loss_fn, val_dataloader, DEVICE)

        wandb.log({"epoch": epoch_i})
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.eval()
            torch.save(model, os.path.join(args.out_dir, model_name+".pt"))
            with open(os.path.join(args.out_dir, model_name+"v.json"), "w") as f:
                json.dump(out_dict, f)
            wandb.log({"min val loss": min_val_loss})

        torch.save(model, os.path.join(args.out_dir, model_name+"last.pt"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses


def gen_caption_from_loader_bart(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
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


        if "," in args.gpu_ids:
            img_feat,img_feat_cls = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat,img_feat_cls = extract_clip_img_feat(model.clip_model, img_tensors)

        src_mask = create_src_mask_bart(src_ids)
        
        ner_mask = torch.ones((args.test_batch_size, args.max_ner_type_len_gt))
        ner_mask = ner_mask.to(DEVICE)

        if "," in args.gpu_ids:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,)
        else:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,)
            else:
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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


def extract_visual_prompt(model, image_features, prompt_mlp_type):
    with torch.no_grad():
        image_features = model.model.prompt_mlp(image_features)
        if prompt_mlp_type == "clipcap":
            image_features = image_features.reshape(image_features.size()[0], args.prompt_size, 768)
        if model.model.embed_dim == 1024:
            image_features = model.model.visual_map(image_features)
    return image_features


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
    set_random_seed(int(args.seed))
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    import torch
    import clip
    from src.data.goodnews_dataset_entity_type_newsmep_ent_ent_pos import GoodNewsDictDatasetEntityTypeFixLenEntPos, collate_fn_goodnews_entity_type

    from venv import create
    
    from numpy import iterable
    
    import json
    from tqdm import tqdm
    
    from src.data.nytimes_dataset_newsmap_ent_article_seg_ent_pos import NYTimesDictDatasetEntityTypeFixLenEntPos, collate_fn_nytimes_entity_type
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup, BartTokenizer, PreTrainedTokenizerFast
    from src.models.modeling_mmbart_clip_inside_vis_clipcap_ent_type_final_fix_len_enc_self_crossattn import BartForMultiModalGeneration
    from torchvision import models, transforms
    import torch.optim as optim
    import re
    import types
    import numpy as np
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    import unidecode
    import wandb


    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = torch.cuda.device_count()
    print(f"world size: {world_size}")
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    DEVICE = torch.device("cuda", local_rank)

    def batch_softmax(phrase_region_match):
        # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
        batch_size, _, num_spans, _ = phrase_region_match.size()

        # [B, B, span_len]
        phrase_region_max = phrase_region_match.max(-1).values

        # Logits [B, B]
        phrase_region_scores = phrase_region_max.sum(-1)
        # Normalize scores
        scale = torch.tensor(num_spans).expand(batch_size).unsqueeze(1).expand((batch_size, batch_size))
        scale = scale.to(phrase_region_scores.device)
        logits = phrase_region_scores.div(scale)

        targets = torch.arange(batch_size).to(logits.device)

        return torch.nn.functional.cross_entropy(logits, targets)
        

    class BatchSoftmax(torch.nn.Module):
        def __init__(self):
            super(BatchSoftmax, self).__init__()

        def forward(self, face_j, ner_j):
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match)
            loss2 = batch_softmax(ner_face_match)
            loss = loss1 + loss2
            return loss
    
    
    plm_type_text = args.plm_type.replace("/", "-")
    clip_type_text = args.clip_type.replace("/", "-")
    ent_start = args.ent_start_token.replace("/", "-")
    ent_end = args.ent_end_token.replace("/", "-")


    if args.only_image:
        if args.trained_clip == "no":
            model_name = args.data_type + f"cross_newsmep_enc-self_onlyvis_retri{args.num_sentences}_CLIP{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}"
        else:
            model_name = args.data_type + f"cross_newsmep_enc-self_onlyvis_retri{args.num_sentences}_trained_CLIP{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}"
            if "_by" in args.trained_clip and "MENM" in args.trained_clip:
                model_name = model_name.replace("trained_CLIP", "trained_CLIP_MENM")
            elif "_by" in args.trained_clip:
                model_name = model_name.replace("trained_CLIP", "trained_CLIP_noMENM")
            else:
                print("old trained CLIP")
    else:
        if args.trained_clip == "no":
            model_name = args.data_type + f"cross_newsmep_enc-self_onlyvis_retri{args.num_sentences}_init{args.init_attn_weight}_CLIP{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}_bos{args.perturb}"
        else:
            model_name = args.data_type + f"cross_newsmep_enc-self_onlyvis_retri{args.num_sentences}_init{args.init_attn_weight}__trained_CLIP{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}_bos{args.perturb}"
            if "_by" in args.trained_clip and "MENM" in args.trained_clip:
                model_name = model_name.replace("trained_CLIP", "trained_CLIP_MENM")
            elif "_by" in args.trained_clip:
                model_name = model_name.replace("trained_CLIP", "trained_CLIP_noMENM")
            else:
                print("old trained CLIP")
    

    if not args.freeze_clip:
        model_name = model_name.replace("freeze", "")
    if args.prompt_mlp_type == "clipcap":
        model_name = model_name.replace(f"{args.map_size}", "")
    elif args.prompt_mlp_type == "mlp":
        model_name = model_name.replace(f"{args.prompt_size}", "")
    
    if args.enc_fusion_layer == [0,1,2,3,4,5] and args.dim_common == 768:
        model_name = model_name.replace(f"{args.enc_fusion_layer}", "all-enc")
    elif args.enc_fusion_layer == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] and args.dim_common == 1024:
        model_name = model_name.replace(f"{args.enc_fusion_layer}", "all-enc")
    elif args.enc_fusion_layer == [0, 1, 2, 3, 4, 5] and args.dim_common == 1024:
        model_name = model_name.replace(f"{args.enc_fusion_layer}", "front-enc")
    elif args.enc_fusion_layer == [6, 7, 8, 9, 10, 11] and args.dim_common == 1024:
        model_name = model_name.replace(f"{args.enc_fusion_layer}", "back-enc")

    
    if args.no_mapping:
        model_name = model_name.replace(f"_map{args.mapping_loss_weight}-{args.mapping_loss_type}","nomap")
    if args.no_clip_norm:
        model_name = model_name.replace(f"-{args.clip_norm}", "")
    
    if not args.do_retrieval:
        model_name = model_name.replace(f"_retri{args.num_sentences}", "")


    model_name = model_name.replace("True", "T")
    model_name = model_name.replace("False", "F")
    model_name = model_name.replace("patrickvonplaten", "")

    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        name=args.experiment_name,
    )

    wandb.config.update(args)

    if args.plm_type.startswith("ainize"):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    else:
        tokenizer = BartTokenizer.from_pretrained(args.plm_type)
    
    if args.trained_clip == "no":
        clip_model, clip_preprocess = clip.load(args.clip_type, device=DEVICE)
    else:
        clip_model = torch.load(os.path.join(args.clip_dir,  args.trained_clip),  map_location=torch.device('cpu'))
        # clip_model = clip_model.to(DEVICE)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    model = BartForMultiModalGeneration.from_pretrained(args.plm_type, enc_fusion_layer=args.enc_fusion_layer, dim_common=args.dim_common, img_size=args.img_size, prompt_mlp_type=args.prompt_mlp_type, map_size=args.map_size, prompt_size=args.prompt_size, clip_model=clip_model, freeze_clip=args.freeze_clip, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, only_image=args.only_image, init_attn_weight=args.init_attn_weight)


    tokenizer.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>"]})
    model.resize_token_embeddings(len(tokenizer))

    if args.perturb:
        bos_noise = torch.randn(1024)
        model.model.shared.weight.data[0] = model.model.shared.weight.data[0] + bos_noise

    del clip_model
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    

    tokenizer_dataset = BartTokenizer.from_pretrained(args.plm_type)
    tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
    
    if args.do_retrieval:
        print("use retrieved sentences")
        if args.data_type == "nytimes":
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            with open(os.path.join(args.data_dir, f"NYTimes/train_dict_newsmep_ent_seg_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                train_dict = json.load(f)        
            train_data = NYTimesDictDatasetEntityTypeFixLenEntPos(train_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)
            train_loader = DataLoader(train_data, args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type, sampler=train_sampler)

            with open(os.path.join(args.data_dir, f"NYTimes/val_dict_newsmep_ent_seg_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                val_dict = json.load(f)
            val_data = NYTimesDictDatasetEntityTypeFixLenEntPos(val_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            val_loader = DataLoader(val_data, args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
            
            with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
            wandb.log({"train size":len(train_data), "val size": len(val_data), "test size": len(test_data)})
        elif args.data_type == "goodnews":
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
            with open(os.path.join(args.data_dir, f"GoodNews/train_dict_newsmep_ent_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                train_dict = json.load(f)
            # print(len(train_dict))
            train_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(train_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)

            train_loader = DataLoader(train_data, args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type, sampler=train_sampler)

            with open(os.path.join(args.data_dir, f"GoodNews/val_dict_newsmep_ent_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                val_dict = json.load(f)
            val_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(val_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            val_loader = DataLoader(val_data, args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)
            
            with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_clip{args.num_sentences}sent_contras_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=True, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)

            wandb.log({"train size":len(train_data), "val size": len(val_data), "test size": len(test_data)})
        else:
            data_base_dir = {} # to be filled for VisualNews
            val_loader = {}
            train_loader = {}
    else:
        if args.data_type == "nytimes":
            data_base_dir = os.path.join(args.data_dir, "NYTimes/nytimes")
            with open(os.path.join(args.data_dir, f"NYTimes/train_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                train_dict = json.load(f)        
            train_data = NYTimesDictDatasetEntityTypeFixLenEntPos(train_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)
            train_loader = DataLoader(train_data, args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type, sampler=train_sampler)

            with open(os.path.join(args.data_dir, f"NYTimes/val_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                val_dict = json.load(f)
            val_data = NYTimesDictDatasetEntityTypeFixLenEntPos(val_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            val_loader = DataLoader(val_data, args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)
            
            with open(os.path.join(args.data_dir, f"NYTimes/test_dict_newsmep_ent_seg_cleaned_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = NYTimesDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_nytimes_entity_type)

            wandb.log({"train size":len(train_data), "val size": len(val_data), "test size": len(test_data)})
        elif args.data_type == "goodnews":
            data_base_dir = os.path.join(args.data_dir, "GoodNews/goodnews")
            with open(os.path.join(args.data_dir, f"GoodNews/train_dict_newsmep_ent_cap_name_pos.json")) as f:
                train_dict = json.load(f)
            # print(len(train_dict))
            train_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(train_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)

            train_loader = DataLoader(train_data, args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type, sampler=train_sampler)

            with open(os.path.join(args.data_dir, f"GoodNews/val_dict_newsmep_ent_cap_name_pos.json")) as f:
                val_dict = json.load(f)
            val_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(val_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            val_loader = DataLoader(val_data, args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)
            
            with open(os.path.join(args.data_dir, f"GoodNews/test_dict_newsmep_ent_cap_name_pos.json")) as f:
                test_dict = json.load(f)
            test_data = GoodNewsDictDatasetEntityTypeFixLenEntPos(test_dict, data_base_dir, tokenizer_dataset, use_clip_tokenizer=True, entity_token_start=args.ent_start_token, entity_token_end=args.ent_end_token, transform = img_transform, max_article_len=args.article_max_length, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, retrieved_sent=False, person_token_id=50267)
            test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_goodnews_entity_type)

            wandb.log({"train size":len(train_data), "val size": len(val_data), "test size": len(test_data)})
        else:
            data_base_dir = {} # to be filled for VisualNews
            val_loader = {}
            train_loader = {}

    model, optimizer_bart, scheduler_bart = prep_for_training(model, len(train_data), DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(DEVICE)

    train(model, loss_fn, train_loader, val_loader, test_loader, optimizer_bart, scheduler_bart, model_name, DEVICE)
    
    
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []

    test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    with open(os.path.join(args.out_dir, model_name+"last.json"), "w") as f:
        json.dump(test_out_dict, f)
    
    wandb.log({"bleu1":blue1, "bleu2":blue2, "bleu3":blue3, "bleu4":blue4, "rouge":rouge_score, "meteor":meteor_score, "cider":cider_score})

    tokenizer.save_pretrained(args.out_dir)
    
