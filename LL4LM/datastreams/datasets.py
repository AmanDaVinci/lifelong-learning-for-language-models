from LL4LM.datastreams.transforms import *

dataset_configs = {
    "udpos": {
        "path": "universal_dependencies",
        "name": "en_lines",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "pan_ner": {
        "path": "wikiann",
        "name": "en",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "few_rel": {
        "path": "few_rel",
        "train_split": "train_wiki",
        "test_split": "val_wiki",
        "transform": few_rel
    },
    "record": {
        "path": "super_glue",
        "name": "record",
        "train_split": "train",
        "test_split": "test",
        "transform": record
    },
    "reviews": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "transform": amazon_reviews
    },
    # multidomain
    "reviews_home": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "home",
        "transform": amazon_reviews
    },
    "reviews_apparel": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "apparel",
        "transform": amazon_reviews
    },
    "reviews_wireless": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "wireless",
        "transform": amazon_reviews
    },
    "reviews_beauty": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "beauty",
        "transform": amazon_reviews
    },
    "reviews_drugstore": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "drugstore",
        "transform": amazon_reviews
    },
    "reviews_kitchen": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "kitchen",
        "transform": amazon_reviews
    },
}