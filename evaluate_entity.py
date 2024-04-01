import os
import json
from tqdm import tqdm
import spacy
from collections import defaultdict
import re
import numpy as np
import types
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import unidecode


def get_proper_nouns(doc):
    proper_nouns = []
    for token in doc:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text.lower())
    return proper_nouns


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text.lower(),
            'label': ent.label_,
            'tokens': [{'text': tok.text.lower(), 'pos': tok.pos_} for tok in ent],
        })
    return entities


def compute_full_recall(caption_names, generated_names):
    count = 0
    for name in caption_names:
        if name in generated_names:
            count += 1
    return count, len(caption_names)


def compute_full_precision(caption_names, generated_names):
    count = 0
    for name in generated_names:
        if name in caption_names:
            count += 1
    return count, len(generated_names)


def compute_entities(caption_entities, gen_entities, c):

    c['n_caption_ents'] += len(caption_entities)
    c['n_gen_ents'] += len(gen_entities)
    for ent in gen_entities:
        if contain_entity(caption_entities, ent):
            c['n_gen_ent_matches'] += 1
    for ent in caption_entities:
        if contain_entity(gen_entities, ent):
            c['n_caption_ent_matches'] += 1

    caption_persons = [e for e in caption_entities if e['label'] == 'PERSON']
    gen_persons = [e for e in gen_entities if e['label'] == 'PERSON']
    c['n_caption_persons'] += len(caption_persons)
    c['n_gen_persons'] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity(caption_persons, ent):
            c['n_gen_person_matches'] += 1
    for ent in caption_persons:
        if contain_entity(gen_persons, ent):
            c['n_caption_person_matches'] += 1

    caption_orgs = [e for e in caption_entities if e['label'] == 'ORG']
    gen_orgs = [e for e in gen_entities if e['label'] == 'ORG']
    c['n_caption_orgs'] += len(caption_orgs)
    c['n_gen_orgs'] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity(caption_orgs, ent):
            c['n_gen_orgs_matches'] += 1
    for ent in caption_orgs:
        if contain_entity(gen_orgs, ent):
            c['n_caption_orgs_matches'] += 1

    caption_gpes = [e for e in caption_entities if e['label'] == 'GPE']
    gen_gpes = [e for e in gen_entities if e['label'] == 'GPE']
    c['n_caption_gpes'] += len(caption_gpes)
    c['n_gen_gpes'] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity(caption_gpes, ent):
            c['n_gen_gpes_matches'] += 1
    for ent in caption_gpes:
        if contain_entity(gen_gpes, ent):
            c['n_caption_gpes_matches'] += 1

    caption_date = [e for e in caption_entities if e['label'] == 'DATE']
    gen_date = [e for e in gen_entities if e['label'] == 'DATE']
    c['n_caption_date'] += len(caption_date)
    c['n_gen_date'] += len(gen_date)
    for ent in gen_date:
        if contain_entity(caption_date, ent):
            c['n_gen_date_matches'] += 1
    for ent in caption_date:
        if contain_entity(gen_date, ent):
            c['n_caption_date_matches'] += 1
    return c


def contain_entity(entities, target):
    for ent in entities:
        if ent['text'] == target['text'] and ent['label'] == target['label']:
            return True
    return False


def contain_entity_by_gtent(entities, target, gt_first=True):
    for ent in entities:
        if gt_first:
            if ent == target['text']:
                return True
        else:
            if ent['text'] == target:
                return True
    return False


def compute_entities_by_gtent(caption_entities, caption_persons, caption_orgs, caption_gpes, gen_entities, c):

    c['n_caption_ents'] += len(caption_entities)
    c['n_gen_ents'] += len(gen_entities)
    for ent in gen_entities:
        if contain_entity_by_gtent(caption_entities, ent, gt_first=True):
            c['n_gen_ent_matches'] += 1
    for ent in caption_entities:
        if contain_entity_by_gtent(gen_entities, ent, gt_first=False):
            c['n_caption_ent_matches'] += 1

    gen_persons = [e for e in gen_entities if e['label'] == 'PERSON']
    c['n_caption_persons'] += len(caption_persons)
    c['n_gen_persons'] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity_by_gtent(caption_persons, ent, gt_first=True):
            c['n_gen_person_matches'] += 1
    for ent in caption_persons:
        if contain_entity_by_gtent(gen_persons, ent, gt_first=False):
            c['n_caption_person_matches'] += 1

    gen_orgs = [e for e in gen_entities if e['label'] == 'ORG']
    c['n_caption_orgs'] += len(caption_orgs)
    c['n_gen_orgs'] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity_by_gtent(caption_orgs, ent, gt_first=True):
            c['n_gen_orgs_matches'] += 1
    for ent in caption_orgs:
        if contain_entity_by_gtent(gen_orgs, ent, gt_first=False):
            c['n_caption_orgs_matches'] += 1

    gen_gpes = [e for e in gen_entities if e['label'] == 'GPE']
    c['n_caption_gpes'] += len(caption_gpes)
    c['n_gen_gpes'] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity_by_gtent(caption_gpes, ent, gt_first=True):
            c['n_gen_gpes_matches'] += 1
    for ent in caption_gpes:
        if contain_entity_by_gtent(gen_gpes, ent, gt_first=False):
            c['n_caption_gpes_matches'] += 1
    return c

def evaluate_entity_by_gtent(output_dict, gtent_dict):
    nlp = spacy.load("en_core_web_lg")
    ent_counter = defaultdict(int)

    counter = 0
    gtent_keys = list(gtent_dict.keys())
    for key,sample in tqdm(output_dict.items()):
        if key not in ["bleu", "other metrics"]:
            gen_cap = nlp(sample["gen"])
            caption_entities = gtent_dict[gtent_keys[counter]]["ner_cap"]
            caption_persons = gtent_dict[gtent_keys[counter]]["names_cap"]
            caption_orgs = gtent_dict[gtent_keys[counter]]["org_cap"]
            caption_gpes = gtent_dict[gtent_keys[counter]]["gpe_cap"]

            gen_entities = get_entities(gen_cap)
            
            compute_entities_by_gtent(caption_entities, caption_persons, caption_orgs, caption_gpes, gen_entities, ent_counter)
            counter += 1

    entity_results = {
        'Entity all - recall': {
            'count': ent_counter['n_caption_ent_matches'],
            'total': ent_counter['n_caption_ents'],
            'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
        },
        'Entity all - precision': {
            'count': ent_counter['n_gen_ent_matches'],
            'total': ent_counter['n_gen_ents'],
            'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
        },
        'Entity person (by full name) - recall': {
            'count': ent_counter['n_caption_person_matches'],
            'total': ent_counter['n_caption_persons'],
            'percentage': ent_counter['n_caption_person_matches'] / ent_counter['n_caption_persons'] if ent_counter['n_caption_persons']> 0 else 0,
        },
        'Entity person (by full name) - precision': {
            'count': ent_counter['n_gen_person_matches'],
            'total': ent_counter['n_gen_persons'],
            'percentage': ent_counter['n_gen_person_matches'] / ent_counter['n_gen_persons'] if ent_counter['n_caption_persons']> 0 else 0,
        },
        'Entity GPE - recall': {
            'count': ent_counter['n_caption_gpes_matches'],
            'total': ent_counter['n_caption_gpes'],
            'percentage': ent_counter['n_caption_gpes_matches'] / ent_counter['n_caption_gpes'],
        },
        'Entity GPE - precision': {
            'count': ent_counter['n_gen_gpes_matches'],
            'total': ent_counter['n_gen_gpes'],
            'percentage': ent_counter['n_gen_gpes_matches'] / ent_counter['n_gen_gpes'],
        },
        'Entity ORG - recall': {
            'count': ent_counter['n_caption_orgs_matches'],
            'total': ent_counter['n_caption_orgs'],
            'percentage': ent_counter['n_caption_orgs_matches'] / ent_counter['n_caption_orgs'],
        },
        'Entity ORG - precision': {
            'count': ent_counter['n_gen_orgs_matches'],
            'total': ent_counter['n_gen_orgs'],
            'percentage': ent_counter['n_gen_orgs_matches'] / ent_counter['n_gen_orgs'],
        },}
    
    output_dict.update(entity_results)
    return entity_results

def evaluate_entity(output_dict):
    nlp = spacy.load("en_core_web_lg")
    # name_recalls, name_precisions = [], []
    full_recall, full_recall_total = 0, 0
    full_precision, full_precision_total = 0, 0
    ent_counter = defaultdict(int)
    for key,sample in tqdm(output_dict.items()):
        if key not in ["bleu", "other metrics"]:
            gt_cap = nlp(sample["gt"])
            gen_cap = nlp(sample["gen"])

            caption_entities = get_entities(gt_cap)
            gen_entities = get_entities(gen_cap)
            names_gt = get_proper_nouns(gt_cap)
            names_gen = get_proper_nouns(gen_cap)
            # print(names_gt, names_gen)
            c, t = compute_full_recall(names_gt, names_gen)
            full_recall += c
            full_recall_total += t

            c, t = compute_full_precision(names_gt, names_gen)
            full_precision += c
            full_precision_total += t

            compute_entities(caption_entities, gen_entities, ent_counter)

    entity_results = {
        'All names (by word) - recall': {
            'count': full_recall,
            'total': full_recall_total,
            'percentage': (full_recall / full_recall_total) if full_recall_total else None,
        },
        'All names (by word) - precision': {
            'count': full_precision,
            'total': full_precision_total,
            'percentage': (full_precision / full_precision_total) if full_precision_total else None,
        },
        'Entity all - recall': {
            'count': ent_counter['n_caption_ent_matches'],
            'total': ent_counter['n_caption_ents'],
            'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
        },
        'Entity all - precision': {
            'count': ent_counter['n_gen_ent_matches'],
            'total': ent_counter['n_gen_ents'],
            'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
        },
        'Entity person (by full name) - recall': {
            'count': ent_counter['n_caption_person_matches'],
            'total': ent_counter['n_caption_persons'],
            'percentage': ent_counter['n_caption_person_matches'] / ent_counter['n_caption_persons'],
        },
        'Entity person (by full name) - precision': {
            'count': ent_counter['n_gen_person_matches'],
            'total': ent_counter['n_gen_persons'],
            'percentage': ent_counter['n_gen_person_matches'] / ent_counter['n_gen_persons'],
        },
        'Entity GPE - recall': {
            'count': ent_counter['n_caption_gpes_matches'],
            'total': ent_counter['n_caption_gpes'],
            'percentage': ent_counter['n_caption_gpes_matches'] / ent_counter['n_caption_gpes'],
        },
        'Entity GPE - precision': {
            'count': ent_counter['n_gen_gpes_matches'],
            'total': ent_counter['n_gen_gpes'],
            'percentage': ent_counter['n_gen_gpes_matches'] / ent_counter['n_gen_gpes'],
        },
        'Entity ORG - recall': {
            'count': ent_counter['n_caption_orgs_matches'],
            'total': ent_counter['n_caption_orgs'],
            'percentage': ent_counter['n_caption_orgs_matches'] / ent_counter['n_caption_orgs'],
        },
        'Entity ORG - precision': {
            'count': ent_counter['n_gen_orgs_matches'],
            'total': ent_counter['n_gen_orgs'],
            'percentage': ent_counter['n_gen_orgs_matches'] / ent_counter['n_gen_orgs'],
        },
        'Entity DATE - recall': {
            'count': ent_counter['n_caption_date_matches'],
            'total': ent_counter['n_caption_date'],
            'percentage': ent_counter['n_caption_date_matches'] / ent_counter['n_caption_date'],
        },
        'Entity DATE - precision': {
            'count': ent_counter['n_gen_date_matches'],
            'total': ent_counter['n_gen_date'],
            'percentage': ent_counter['n_gen_date_matches'] / ent_counter['n_gen_date'],
        },}
    
    # output_dict.update(entity_results)
    return entity_results


def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


def cal_caption_score_from_dict(result_dict):
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)

    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    meteor_scores = []
    for sample in enumerate(tqdm(result_dict.values())):
        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', sample[1]["gt"])
        generation = re.sub(r'[^\w\s]', '', sample[1]["gen"])

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()

    return blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score


def split_dict_by_face_group(testdata_dict, result_dict, data_type="goodnews"):  
    data_keys = list(testdata_dict.keys())
    dict_noface_noname = {}
    dict_face_noname = {}
    dict_noface_name = {}
    dict_face_name = {}
    counter = 0
    for key, value in tqdm(result_dict.items()):
        # take only the sentence pairs, not the metrics
        if counter <len(result_dict)-14:
            if data_type == "goodnews":
                name_gt = testdata_dict[data_keys[int(key)]]["names"]
            else:
                name_gt = testdata_dict[data_keys[int(key)]]["name_cap"]
            face_dir = testdata_dict[data_keys[int(key)]]["face_emb_dir"]
            if type(face_dir) is list and len(name_gt) == 0:
                dict_noface_noname[key] = {}
                dict_noface_noname[key] = value
            elif type(face_dir) is str and len(name_gt) == 0:
                dict_face_noname[key] = {}
                dict_face_noname[key] = value
            elif type(face_dir) is list and len(name_gt) > 0:
                dict_noface_name[key] = {}
                dict_noface_name[key] = value
            else:
                dict_face_name[key] = {}
                dict_face_name[key] = value
            counter +=1
        else:
            break
    print(f"length of noface noname:{len(dict_noface_noname)} | face noname:{len(dict_face_noname)} | noface name:{len(dict_noface_name)} | face name:{len(dict_face_name)}")
    return dict_noface_noname, dict_face_noname, dict_noface_name, dict_face_name


def cal_scores_from_dict(result_dict):
    caption_scores = cal_caption_score_from_dict(result_dict)
    # entity_scores = evaluate_entity(result_dict)
    return{
        # "entity scores": entity_scores,
        "caption scores": caption_scores,
    }

def cal_scores_by_face_group(dict_noface_noname, dict_face_noname, dict_noface_name, dict_face_name):
    scores_noface_noname = cal_scores_from_dict(dict_noface_noname)
    print("No face, no name:", scores_noface_noname)
    # scores_face_noname = cal_scores_from_dict(dict_face_noname)
    scores_noface_name = cal_scores_from_dict(dict_noface_name)
    print("No face, name:", scores_noface_name)
    scores_face_name = cal_scores_from_dict(dict_face_name)
    print("face, name:", scores_face_name)




def split_dict_by_face_group_gtent(testdata_dict, result_dict, gtent_dict, data_type="goodnews"):  
    data_keys = list(testdata_dict.keys())
    dict_noface_noname = {}
    dict_noface_noname_gtent = {}
    dict_face_noname = {}
    dict_face_noname_gtent = {}
    dict_noface_name = {}
    dict_noface_name_gtent = {}
    dict_face_name = {}
    dict_face_name_gtent = {}
    counter = 0
    for key, value in tqdm(result_dict.items()):
        # take only the sentence pairs, not the metrics
        if counter <len(result_dict)-14:
        # if counter <len(result_dict)-2:
            if data_type == "goodnews":
                name_gt = testdata_dict[data_keys[int(key)]]["names"]
            else:
                name_gt = testdata_dict[data_keys[int(key)]]["name_cap"]
            face_dir = testdata_dict[data_keys[int(key)]]["face_emb_dir"]
            if type(face_dir) is list and len(name_gt) == 0:
                dict_noface_noname[key] = {}
                dict_noface_noname[key] = value
                dict_noface_noname_gtent[key] = {}
                dict_noface_noname_gtent[key] = gtent_dict[data_keys[int(key)]]
            elif type(face_dir) is str and len(name_gt) == 0:
                dict_face_noname[key] = {}
                dict_face_noname[key] = value
                dict_face_noname_gtent[key] = {}
                dict_face_noname_gtent[key] = gtent_dict[data_keys[int(key)]]
            elif type(face_dir) is list and len(name_gt) > 0:
                dict_noface_name[key] = {}
                dict_noface_name[key] = value
                dict_noface_name_gtent[key] = {}
                dict_noface_name_gtent[key] = gtent_dict[data_keys[int(key)]]
            else:
                dict_face_name[key] = {}
                dict_face_name[key] = value
                dict_face_name_gtent[key] = {}
                dict_face_name_gtent[key] = gtent_dict[data_keys[int(key)]]
            counter +=1
        else:
            break
    print(f"length of noface noname:{len(dict_noface_noname), len(dict_noface_noname_gtent)} | face noname:{len(dict_face_noname), len(dict_face_noname_gtent)} | noface name:{len(dict_noface_name), len(dict_noface_name_gtent)} | face name:{len(dict_face_name), len(dict_face_name_gtent)}")
    return dict_noface_noname, dict_noface_noname_gtent, dict_face_noname, dict_face_noname_gtent, dict_noface_name, dict_noface_name_gtent, dict_face_name, dict_face_name_gtent


def cal_scores_by_face_group_gtent(dict_noface_noname, dict_noface_noname_gtent, dict_face_noname, dict_face_noname_gtent, dict_noface_name, dict_noface_name_gtent, dict_face_name, dict_face_name_gtent):
    scores_noface_noname = evaluate_entity_by_gtent(dict_noface_noname, dict_noface_noname_gtent)
    print("No face, no name:", scores_noface_noname)
    # scores_face_noname = cal_scores_from_dict(dict_face_noname)
    scores_noface_name = evaluate_entity_by_gtent(dict_noface_name, dict_noface_name_gtent)
    print("No face, name:", scores_noface_name)
    scores_face_name = evaluate_entity_by_gtent(dict_face_name, dict_face_name_gtent)
    print("face, name:", scores_face_name)


def get_nytimes_dict_gtent(test_dict_old, test_dict_with_ent):
    test_dict_gtent = {}
    all_keys = list(test_dict_with_ent.keys())
    for key in test_dict_old.keys():
        if key in all_keys:
            test_dict_gtent[key] = {}
            test_dict_gtent[key]["names_cap"] = [unidecode.unidecode(names.lower()) for names in test_dict_with_ent[key]["name_cap"]]
            test_dict_gtent[key]["org_cap"] = [unidecode.unidecode(names.lower()) for names in test_dict_with_ent[key]["org_cap"]]
            test_dict_gtent[key]["gpe_cap"] = [unidecode.unidecode(names.lower()) for names in test_dict_with_ent[key]["gpe_cap"]]
            test_dict_gtent[key]["ner_cap"] = [unidecode.unidecode(names.lower()) for names in test_dict_with_ent[key]["ner_cap"]]
    return test_dict_gtent
    
if __name__ == "__main__":
    with open("/DATADIR/GoodNews/test_dict_raw.json") as f:
        testdata_dict = json.load(f)

    with open("/OUTDIR/OUTJSON.json") as f:
        result_dict = json.load(f)

    with open("/DATADIR/GoodNews/test_dict_capent.json") as f:
        test_dict_capent = json.load(f)
    
    # replace 23053 with 21537 for NYTimes800k
    print(evaluate_entity_by_gtent({k: result_dict[k] for k in list(result_dict)[:23053]}, test_dict_capent))

    dict_noface_noname, dict_noface_noname_gtent, dict_face_noname, dict_face_noname_gtent, dict_noface_name, dict_noface_name_gtent, dict_face_name, dict_face_name_gtent = split_dict_by_face_group_gtent(testdata_dict, result_dict, test_dict_capent, data_type="goodnews")

    cal_scores_by_face_group_gtent(dict_noface_noname, dict_noface_noname_gtent, dict_face_noname, dict_face_noname_gtent, dict_noface_name, dict_noface_name_gtent, dict_face_name, dict_face_name_gtent)

    # caption scores
    cal_scores_by_face_group(dict_noface_noname, dict_face_noname, dict_noface_name, dict_face_name)
