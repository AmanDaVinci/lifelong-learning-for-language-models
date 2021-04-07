import random
from functools import wraps
from datasets import load_dataset
from LL4LM.datastreams.transforms import DatastreamTransforms
from LL4LM.datastreams.dataset_ids import DatasetID
from LL4LM.datastreams.dataset_ids import (
    SUPER_GLUE, 
    SUPER_GLUE_TEST, 
    AMAZON_REVIEWS,
    AMAZON_REVIEWS_TEST,
    XNLI, 
    XNLI_TEST, 
    mnli_train_id,
    mnli_test_id,
)


@wraps(load_dataset)
def load_datastream(path, name, split, filter_column="", filter_value="", **kwargs):
    ''' datasets.load_dataset wrapper function to return a lifelong datastream '''
    dataset_id = DatasetID(path, name, split, filter_column, filter_value)
    transform = DatastreamTransforms.get(dataset_id)
    dataset = load_dataset(path, name, split=split, **kwargs)
    if filter_column and filter_value:
        dataset = dataset.filter(lambda batch: batch[filter_column]==filter_value)
    datastream = dataset.map(transform, batched=True, remove_columns=dataset.column_names)
    try:
        datastream = datastream.cast(DatastreamTransforms.features)
    except:
        raise ValueError(f"{transform} doesn't correctly transform to datastream features.")
    return datastream

def load_dataset_ids(multitask: bool=True, 
                     multilingual: bool=False, 
                     multidomain: bool=False,
                     shuffle: bool=False):
    dataset_ids, testset_ids  = [], []
    pretrain_dataset_ids, pretrain_testset_ids = [], []
    if multitask:
        dataset_ids.extend(SUPER_GLUE)
        testset_ids.extend(SUPER_GLUE_TEST)
    if multilingual:
        pretrain_dataset_ids.append(mnli_train_id)
        pretrain_testset_ids.append(mnli_test_id)
        dataset_ids.extend(XNLI)
        testset_ids.extend(XNLI_TEST)
    if multidomain:
        dataset_ids.extend(AMAZON_REVIEWS)
        testset_ids.extend(AMAZON_REVIEWS_TEST)
    if shuffle:
        random.shuffle(dataset_ids)
    dataset_ids = pretrain_dataset_ids+dataset_ids
    testset_ids = pretrain_testset_ids+testset_ids
    return dataset_ids, testset_ids