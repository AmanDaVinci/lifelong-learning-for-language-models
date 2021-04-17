# TODO: All dataset specific functions can be refactored into task specific functions

import random
from datasets import Features, Value, ClassLabel
from LL4LM.datastreams.dataset_ids import (
    DatasetID,
    boolq_train_id,
    boolq_eval_id,
    boolq_test_id,
    multirc_train_id,
    multirc_eval_id,
    multirc_test_id,
    cb_train_id,
    cb_eval_id,
    cb_test_id,
    copa_train_id,
    copa_eval_id,
    copa_test_id,
    record_train_id,
    record_eval_id,
    record_test_id,
    rte_train_id,
    rte_eval_id,
    rte_test_id,
    wsc_train_id,
    wsc_eval_id,
    wsc_test_id,
    wic_train_id,
    wic_eval_id,
    wic_test_id,
    mnli_train_id,
    mnli_eval_id,
    mnli_test_id,
    xnli_eval_id,
    xnli_test_id,
    xnli_eval_ar_id,
    xnli_eval_bg_id,
    xnli_eval_de_id,
    xnli_eval_el_id,
    xnli_eval_en_id,
    xnli_eval_es_id,
    xnli_eval_fr_id,
    xnli_eval_hi_id,
    xnli_eval_ru_id,
    xnli_eval_sw_id,
    xnli_eval_th_id,
    xnli_eval_tr_id,
    xnli_eval_ur_id,
    xnli_eval_vi_id,
    xnli_eval_zh_id,
    xnli_test_ar_id,
    xnli_test_bg_id,
    xnli_test_de_id,
    xnli_test_el_id,
    xnli_test_en_id,
    xnli_test_es_id,
    xnli_test_fr_id,
    xnli_test_hi_id,
    xnli_test_ru_id,
    xnli_test_sw_id,
    xnli_test_th_id,
    xnli_test_tr_id,
    xnli_test_ur_id,
    xnli_test_vi_id,
    xnli_test_zh_id,
    amazon_reviews_home_id,
    amazon_reviews_apparel_id,
    amazon_reviews_wireless_id,
    amazon_reviews_beauty_id,
    amazon_reviews_drugstore_id,
    amazon_reviews_kitchen_id,
    amazon_reviews_test_home_id,
    amazon_reviews_test_apparel_id,
    amazon_reviews_test_wireless_id,
    amazon_reviews_test_beauty_id,
    amazon_reviews_test_drugstore_id,
    amazon_reviews_test_kitchen_id,
    fewrel_train_id,
    fewrel_test_id,
    wikiann_train_id,
    wikiann_test_id,
    udpos_train_id,
    udpos_test_id,
)


def boolq(batch: dict) -> dict:
    return {
        "context": batch["passage"],
        "statement": batch["question"],
        "label": batch["label"],
    }

def cb(batch: dict) -> dict:
    label2string = {0:'entailment', 1:'contradiction', 2:'neutral'}
    task_descriptors = [". This implies ", ". This is "]
    contexts, statements, labels = [], [], []
    for row in zip(batch["premise"], 
                   batch["hypothesis"],
                   batch["label"]):
        context, hypothesis, label = row
        contexts.append(context)
        desc = random.choice(task_descriptors)
        statements.append(" ".join([hypothesis, desc, label2string[label]]))
        labels.append(1)
        for other_label, other_label_str in label2string.items():
            if other_label != label:
                contexts.append(context)
                statements.append(" ".join([hypothesis, desc, other_label_str]))
                labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def copa(batch: dict) -> dict:
    contexts, statements, labels = [], [], []
    for row in zip(batch["premise"], 
                   batch["question"],
                   batch["choice1"],
                   batch["choice2"],
                   batch["label"]):
        context, question, choice0, choice1, correct_choice = row

        contexts.append(context)
        statements.append(" ".join([question, choice0]))
        labels.append(1 if correct_choice==0 else 0)

        contexts.append(context)
        statements.append(" ".join([question, choice1]))
        labels.append(1 if correct_choice==1 else 0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def multirc(batch: dict) -> dict:
    return {
        "context": batch["paragraph"],
        "statement":  [" ".join([q, a]) \
            for q, a in zip(batch["question"], batch["answer"])],
        "label": batch["label"]
    }

def record(batch: dict) -> dict:
    contexts, statements, labels = [], [], []
    for row in zip(batch["passage"], 
                    batch["query"],
                    batch["entities"],
                    batch["answers"]):
        context, query, choices, answers = row
        for choice in choices:
            contexts.append(context)
            statements.append(query.replace("@placeholder", choice))
            labels.append(1 if choice in answers else 0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }
        
def rte(batch: dict) -> dict:
    label2string = {0:'entailment', 1:'not_entailment'}
    task_descriptors = [". This implies ", ". This is "]
    contexts, statements, labels = [], [], []
    for row in zip(batch["premise"], 
                   batch["hypothesis"],
                   batch["label"]):
        context, hypothesis, label = row
        contexts.append(context)
        desc = random.choice(task_descriptors)
        statements.append(" ".join([hypothesis, desc, label2string[label]]))
        labels.append(1)
        for other_label, other_label_str in label2string.items():
            if other_label != label:
                contexts.append(context)
                statements.append(" ".join([hypothesis, desc, other_label_str]))
                labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def wic(batch: dict) -> dict:
    task_descriptors = [
        " is the polysemous word.", 
        " is used with the same sense."
    ]
    contexts = [" ".join([sen1, sen2]) \
                for sen1, sen2 in zip(batch["sentence1"], batch["sentence2"])]
    desc = random.choice(task_descriptors)
    statements = [" ".join([word, desc]) for word in batch["word"]]
    return {
        "context": contexts,
        "statement": statements,
        "label": batch["label"]
    }

def wsc(batch: dict) -> dict:
    task_descriptors = [
        " refers to ", 
        " is ", 
        " is the pronoun of "
    ]
    contexts = batch["text"]
    desc = random.choice(task_descriptors)
    statements = [" ".join([pronoun, desc, noun]) \
                    for noun, pronoun in zip(batch["span1_text"], batch["span2_text"])]
    return {
        "context": contexts,
        "statement": statements,
        "label": batch["label"]
    }

def mnli(batch: dict) -> dict:
    label2string = {0:'entailment', 1:'neutral', 2:'contradiction'}
    task_descriptors = [". This implies ", ". This is "]
    contexts, statements, labels = [], [], []
    for row in zip(batch["premise"], 
                   batch["hypothesis"],
                   batch["label"]):
        context, hypothesis, label = row
        contexts.append(context)
        desc = random.choice(task_descriptors)
        statements.append(" ".join([hypothesis, desc, label2string[label]]))
        labels.append(1)
        for other_label, other_label_str in label2string.items():
            if other_label != label:
                contexts.append(context)
                statements.append(" ".join([hypothesis, desc, other_label_str]))
                labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def xnli(batch: dict) -> dict:
    all_labels = ('entailment', 'neutral', 'contradiction')
    task_descriptors = [". This implies ", ". This is "]
    contexts, statements, labels = [], [], []
    for row in zip(batch["sentence1"], 
                   batch["sentence2"],
                   batch["gold_label"]):
        context, hypothesis, label = row
        contexts.append(context)
        desc = random.choice(task_descriptors)
        statements.append(" ".join([hypothesis, desc, label]))
        labels.append(1)
        for other_label in all_labels:
            if other_label != label:
                contexts.append(context)
                statements.append(" ".join([hypothesis, desc, other_label]))
                labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def amazon_reviews(batch: dict) -> dict:
    label2string = {
        1:'negative',
        2:'negative', 
        3:'neutral',
        4:'positive',
        5:'positive',
    }
    contexts, statements, labels = [], [], []
    for row in zip(batch["review_title"], 
                   batch["review_body"],
                   batch["stars"]):
        review_title, review_body, label_int = row
        contexts.append(" ".join([review_title, review_body]))
        label_str = label2string[label_int]
        statement = random.choice([
            f"It is a {label_str} review.",
            f"The sentiment is {label_str}.",
        ])
        statements.append(statement)
        labels.append(1)
        for other_label_int, other_label_str in label2string.items():
            if other_label_str != label_str:
                contexts.append(" ".join([review_title, review_body]))
                statement = random.choice([
                    f"It is a {other_label_str} review.",
                    f"The sentiment is {other_label_str}.",
                ])
                statements.append(statement)
                labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def few_rel(batch: dict) -> dict:
    '''TODO: select relevant false relations to increase difficulty'''
    num_false_statements = 3
    relations = ['place served by transport hub', 'mountain range', 'religion', 'participating team', 'contains administrative territorial entity', 'head of government', 'country of citizenship', 'original network', 'heritage designation', 'performer', 'participant of', 'position held', 'has part', 'location of formation', 'located on terrain feature', 'architect', 'country of origin', 'publisher', 'director', 'father', 'developer', 'military branch', 'mouth of the watercourse', 'nominated for', 'movement', 'successful candidate', 'followed by', 'manufacturer', 'instance of', 'after a work by', 'member of political party', 'licensed to broadcast to', 'headquarters location', 'sibling', 'instrument', 'country', 'occupation', 'residence', 'work location', 'subsidiary', 'participant', 'operator', 'characters', 'occupant', 'genre', 'operating system', 'owned by', 'platform', 'tributary', 'winner', 'said to be the same as', 'composer', 'league', 'record label', 'distributor', 'screenwriter', 'sports season of league or competition', 'taxon rank', 'location', 'field of work', 'language of work or name', 'applies to jurisdiction', 'notable work', 'located in the administrative territorial entity']
    contexts, statements, labels = [], [], []
    for row in zip(batch["tokens"], 
                   batch["names"],
                   batch["head"],
                   batch["tail"]):
        context = " ".join(row[0])
        relation = row[1][0]
        head, tail = row[2]["text"], row[3]["text"]
        true_statement = " - ".join([head, relation, tail]) 
        contexts.append(context)
        statements.append(true_statement)
        labels.append(1)
        remaining_relations = [rel for rel in relations if rel!=relation]
        for false_relation in random.sample(remaining_relations, k=num_false_statements):
            false_statement = " - ".join([head, false_relation, tail]) 
            contexts.append(context)
            statements.append(false_statement)
            labels.append(0)
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def wikiann(batch: dict) -> dict:
    num_false_statements = 3
    corruption_probability = 0.5
    #named_entities = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    named_entities = ['Outside', 'Beginning-Person', 'Inside-Person', 'Beginning-Organization', 'Inside-Organization', 'Beginning-Location', 'Inside-Location']
    task_descriptors = ["Name of the entities: ", "Entity names: "]
    contexts, statements, labels = [], [], []
    for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):
        context = " ".join(tokens)
        tags = [named_entities[tag] for tag in tags]
        desc = random.choice(task_descriptors)
        true_statement = " ".join([desc]+tags)
        contexts.append(context)
        statements.append(true_statement)
        labels.append(1)
        seq_len = len(tags)
        for _ in range(num_false_statements):
            indices_to_corrupt = random.sample(range(seq_len), k=int(corruption_probability*seq_len))
            corrupted_tags = [
                random.choice(named_entities) if i in indices_to_corrupt else tag\
                for i, tag in enumerate(tags)
            ]
            false_statement = " ".join([desc]+corrupted_tags)
            contexts.append(context)
            statements.append(false_statement)
            labels.append(0)
            
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

def udpos(batch: dict) -> dict:
    num_false_statements = 3
    corruption_probability = 0.5
    pos_list = ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', '_', 'ADV', 'INTJ', 'VERB', 'AUX']
    task_descriptors = ["Part of speech tags: ", "POS tags: "]
    contexts, statements, labels = [], [], []
    for tokens, tags in zip(batch["tokens"], batch["upos"]):
        context = " ".join(tokens)
        tags = [pos_list[tag] for tag in tags]
        desc = random.choice(task_descriptors)
        true_statement = " ".join([desc]+tags)
        contexts.append(context)
        statements.append(true_statement)
        labels.append(1)
        seq_len = len(tags)
        for _ in range(num_false_statements):
            indices_to_corrupt = random.sample(range(seq_len), k=int(corruption_probability*seq_len))
            corrupted_tags = [
                random.choice(pos_list) if i in indices_to_corrupt else tag\
                for i, tag in enumerate(tags)
            ]
            false_statement = " ".join([desc]+corrupted_tags)
            contexts.append(context)
            statements.append(false_statement)
            labels.append(0)
            
    return {
        "context": contexts,
        "statement": statements,
        "label": labels
    }

class DatastreamTransforms:
    transforms = {
        boolq_train_id: boolq,
        boolq_eval_id: boolq,
        boolq_test_id: boolq,

        multirc_train_id: multirc,
        multirc_eval_id: multirc,
        multirc_test_id: multirc,

        cb_train_id: cb,
        cb_eval_id: cb,
        cb_test_id: cb,

        copa_train_id: copa,
        copa_eval_id: copa,
        copa_test_id: copa,

        record_train_id: record,
        record_eval_id: record,
        record_test_id: record,

        rte_train_id: rte,
        rte_eval_id: rte,
        rte_test_id: rte,

        wsc_train_id: wsc,
        wsc_eval_id: wsc,
        wsc_test_id: wsc,

        wic_train_id: wic,
        wic_eval_id: wic,
        wic_test_id: wic,

        mnli_train_id: mnli,
        mnli_eval_id: mnli,
        mnli_test_id: mnli,

        xnli_eval_id: xnli,
        xnli_test_id: xnli,

        xnli_eval_ar_id: xnli,
        xnli_eval_bg_id: xnli,
        xnli_eval_de_id: xnli,
        xnli_eval_el_id: xnli,
        xnli_eval_en_id: xnli,
        xnli_eval_es_id: xnli,
        xnli_eval_fr_id: xnli,
        xnli_eval_hi_id: xnli,
        xnli_eval_ru_id: xnli,
        xnli_eval_sw_id: xnli,
        xnli_eval_th_id: xnli,
        xnli_eval_tr_id: xnli,
        xnli_eval_ur_id: xnli,
        xnli_eval_vi_id: xnli,
        xnli_eval_zh_id: xnli,

        xnli_test_ar_id: xnli,
        xnli_test_bg_id: xnli,
        xnli_test_de_id: xnli,
        xnli_test_el_id: xnli,
        xnli_test_en_id: xnli,
        xnli_test_es_id: xnli,
        xnli_test_fr_id: xnli,
        xnli_test_hi_id: xnli,
        xnli_test_ru_id: xnli,
        xnli_test_sw_id: xnli,
        xnli_test_th_id: xnli,
        xnli_test_tr_id: xnli,
        xnli_test_ur_id: xnli,
        xnli_test_vi_id: xnli,
        xnli_test_zh_id: xnli,

        amazon_reviews_home_id: amazon_reviews,
        amazon_reviews_apparel_id: amazon_reviews,
        amazon_reviews_wireless_id: amazon_reviews,
        amazon_reviews_beauty_id: amazon_reviews,
        amazon_reviews_drugstore_id: amazon_reviews,
        amazon_reviews_kitchen_id: amazon_reviews,

        amazon_reviews_test_home_id: amazon_reviews,
        amazon_reviews_test_apparel_id: amazon_reviews,
        amazon_reviews_test_wireless_id: amazon_reviews,
        amazon_reviews_test_beauty_id: amazon_reviews,
        amazon_reviews_test_drugstore_id: amazon_reviews,
        amazon_reviews_test_kitchen_id: amazon_reviews,

        fewrel_train_id: few_rel,
        fewrel_test_id: few_rel,

        wikiann_train_id: wikiann,
        wikiann_test_id: wikiann,

        udpos_train_id: udpos,
        udpos_test_id: udpos,
    }
    # All transforms must return a batch with the following Features definition
    features = Features({
        "context":      Value("string"),
        "statement":    Value("string"),
        "label":        ClassLabel(2, names=["False", "True"])
    })

    @classmethod
    def get(cls, dataset_id: DatasetID):
        if dataset_id in cls.transforms:
            return cls.transforms[dataset_id]
        else:
            raise NotImplementedError(f"{dataset_id} transform is not implemented.")