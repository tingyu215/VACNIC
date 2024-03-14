#!/bin/sh
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port 29501 /CODEDIR/train_mmbart_enc_self_face_name_ids_retrieve_crossattn_bart_guide_match.py --seed 684331 \
--gpu_ids 2,3 --num_workers 16 \
--article_max_length 512 --caption_max_length 100 \
--plm_type facebook/bart-base \
--clip_type ViT-B/16 \
--ent_start_token "<ENT>" --ent_end_token "<ENT>" \
--enc_fusion_layer 0 1 2 3 4 5 \
--dim_common 768 \
--warmup_rate 0.05 --train_batch_size 12 --val_batch_size 1 --test_batch_size 1 \
--beam_size 5 --max_length 50 \
--num_epoch 16 --lr_bart 3e-5 --lr_clip 1e-7 \
--weight_decay 0.01 --clip_norm 0.1 \
--no_clip_norm True \
--data_type goodnews \
--data_dir /DATADIR \
--out_dir /OUTPUTDIR \
--mapping_loss_type contrastive \
--trained_clip CLIPNAME.pt \
--clip_dir /CLIPDIR \
--no_clip_loss True \
--prompt_size 20 --use_vis_cls True \
--max_ner_type_len 80 --max_ner_type_len_gt 20 \
--freeze_clip True \
--prompt_mlp_type clipcap --map_size 196 256 64 16 \
--no_mapping False --mapping_loss_weight 1 \
--img_size 768 \
--only_image False \
--use_secla True \
--num_sentences 8 \
--project_name news_cap \
--experiment_name crossattn_retrieval_goodnews_bart_guide_match \
--offline_wandb False \
--perturb False \
--init_attn_weight False \
--margin 1.0 \
--alpha 0.5
