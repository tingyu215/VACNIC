# VACNIC
Official code for [Visually-Aware Context Modeling for News Image Captioning](https://arxiv.org/abs/2308.08325) (NAACL 2024)

* ~~Full code will be uploaded (and cleaned) soon.~~
* ~~The new version of the paper will be updated soon.~~
* The major part of the codebase is complete, I will add missing things from time to time (if any)

## Installation
To create conda env, please run:

    conda env create -n vacnic --file vacnic.yml
    conda activate vacnic

## Data
Please download GoodNews and NYTimes800k datasets from the official repo of [Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell).
In our implementation, we use json files for loading the datasets. Please find the processed json files via [this link](https://drive.google.com/drive/folders/1BcvSrpN5V_qkhpGeDrZTcXdOMWlTj_MB?usp=share_link)
After downloading, please change all **DATADIR** in json files to the dir where you store your datasets.

\*The code for generating such json files will be cleaned



## Training

To train the full model, run:

    run_full_train.sh

The shell script shows how to run the BART-large version of our model (which is trained on one A100 GPU for roughly 1.5 days).
Before running the script, please change the following commands first:

    CODEDIR --> the path to where you store the codebase
    DATADIR --> the path to where you store the datasets
    OUTPUTDIR --> the path to where you store the output models
    CLIPNAME/CLIPDIR* --> the name/dir for the tuned CLIP (if any). 

\*If no tuned CLIP is used, call *--trained_clip no*. Note that we found that tuning CLIP on the NewsCap datasets does not bring significant improvements in performance.

During training, we use a ClipCap style of mapping for visual features:
    
    --prompt_mlp_type clipcap

The command *--map_size* is only useful when you are using non-clipcap style mappings.

For Face Naming loss, as proposed in our [SECLA paper (WACV2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Qu_Weakly_Supervised_Face_Naming_With_Symmetry-Enhanced_Contrastive_Loss_WACV_2023_paper.pdf), set

    --use_secla True
SECLA is quite useful if you want to learn alignment in a weakly supervised way. SECLA-B is a curriculum learning variant of SECLA that allows you to learn from easy to difficult examples. Our code for SECLA is available [here](https://github.com/tingyu215/SECLA)

The hyperparameters for CoLaM are:

    --margin 1.0 
    --alpha 0.5

where alpha is $\alpha$, margin is $\Delta$.


We also provide example code for training model with only visual prompt for reference (one can also do so by setting *--only_image* in run_full_train.sh to True). To run this simple baseline, run run_onlyvis_trian.sh, in which *--do_retrieval* controls whether to use retrieved article segments for training.


## Evaluation

By the end of our training code, we automatically generate the json file containing generated captions & ground truth captions. The caption generateion scores will be printed out and stored in wandb log.

For evaluating entities, please run
    
    python evaluate_entity.py

where 

    DATADIR is the root dir for the datasets
    OUTDIR is the output dir of your json file

Note that the package version needs to be changed, please refer to the repo of [Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell) (also where you get the raw datasets from), and use the versions indicated in their repo.


## Prompt with LMM

We use the following environment to run prompting with large multimodal models:

    conda env create -n pytorch20 --file pytorch20.yml
    conda activate pytorch20

Run run_test_instructblip_prompt.sh or run_test_llava_prompt.sh for the experiment

    --use_retrieval True/False
this line controls whether to use CLIP retrieved article segments


## Citation

    @article{qu2023visuallyaware,
      title={Visually-Aware Context Modeling for News Image Captioning}, 
      author={Tingyu Qu and Tinne Tuytelaars and Marie-Francine Moens},
      journal={arXiv preprint arXiv:2308.08325},
      year={2023}
    }