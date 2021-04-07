from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class DatasetID:
    path: str
    name: str
    split: str
    filter_column: str = ""
    filter_value: str = ""

    def __str__(self) -> str:
        if self.filter_column and self.filter_value:
            return f"{self.path}.{self.name}.{self.filter_column}:{self.filter_value}"
        else:
            return f"{self.path}.{self.name}"

boolq_train_id = DatasetID("super_glue", "boolq", "train")
boolq_eval_id = DatasetID("super_glue", "boolq", "validation")
boolq_test_id = DatasetID("super_glue", "boolq", "test")

multirc_train_id = DatasetID("super_glue", "multirc", "train")
multirc_eval_id = DatasetID("super_glue", "multirc", "validation")
multirc_test_id = DatasetID("super_glue", "multirc", "test")

cb_train_id = DatasetID("super_glue", "cb", "train")
cb_eval_id = DatasetID("super_glue", "cb", "validation")
cb_test_id = DatasetID("super_glue", "cb", "test")

copa_train_id = DatasetID("super_glue", "copa", "train")
copa_eval_id = DatasetID("super_glue", "copa", "validation")
copa_test_id = DatasetID("super_glue", "copa", "test")

record_train_id = DatasetID("super_glue", "record", "train")
record_eval_id = DatasetID("super_glue", "record", "eval")
record_test_id = DatasetID("super_glue", "record", "test")

rte_train_id = DatasetID("super_glue", "rte", "train")
rte_eval_id = DatasetID("super_glue", "rte", "eval")
rte_test_id = DatasetID("super_glue", "rte", "test")

wic_train_id = DatasetID("super_glue", "wic", "train")
wic_eval_id = DatasetID("super_glue", "wic", "eval")
wic_test_id = DatasetID("super_glue", "wic", "test")

wsc_train_id = DatasetID("super_glue", "wsc", "train")
wsc_eval_id = DatasetID("super_glue", "wsc", "eval")
wsc_test_id = DatasetID("super_glue", "wsc", "test")

mnli_train_id = DatasetID("glue", "mnli", "train")
mnli_eval_id = DatasetID("glue", "mnli", "validation_matched")
mnli_test_id = DatasetID("glue", "mnli", "test_matched")

xnli_eval_id = DatasetID("xtreme", "XNLI", "validation")
xnli_test_id = DatasetID("xtreme", "XNLI", "test")

xnli_eval_ar_id = DatasetID("xtreme", "XNLI", "validation", "language", "ar")
xnli_eval_bg_id = DatasetID("xtreme", "XNLI", "validation", "language", "bg")
xnli_eval_de_id = DatasetID("xtreme", "XNLI", "validation", "language", "de")
xnli_eval_el_id = DatasetID("xtreme", "XNLI", "validation", "language", "el")
xnli_eval_en_id = DatasetID("xtreme", "XNLI", "validation", "language", "en")
xnli_eval_es_id = DatasetID("xtreme", "XNLI", "validation", "language", "es")
xnli_eval_fr_id = DatasetID("xtreme", "XNLI", "validation", "language", "fr")
xnli_eval_hi_id = DatasetID("xtreme", "XNLI", "validation", "language", "hi")
xnli_eval_ru_id = DatasetID("xtreme", "XNLI", "validation", "language", "ru")
xnli_eval_sw_id = DatasetID("xtreme", "XNLI", "validation", "language", "sw")
xnli_eval_th_id = DatasetID("xtreme", "XNLI", "validation", "language", "th")
xnli_eval_tr_id = DatasetID("xtreme", "XNLI", "validation", "language", "tr")
xnli_eval_ur_id = DatasetID("xtreme", "XNLI", "validation", "language", "ur")
xnli_eval_vi_id = DatasetID("xtreme", "XNLI", "validation", "language", "vi")
xnli_eval_zh_id = DatasetID("xtreme", "XNLI", "validation", "language", "zh")

amazon_reviews_home_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "home")
amazon_reviews_apparel_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "apparel")
amazon_reviews_wireless_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "wireless")
amazon_reviews_beauty_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "beauty")
amazon_reviews_drugstore_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "drugstore")
amazon_reviews_kitchen_id = DatasetID("amazon_reviews_multi", "en", "train", "product_category", "kitchen")
 
amazon_reviews_test_home_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "home")
amazon_reviews_test_apparel_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "apparel")
amazon_reviews_test_wireless_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "wireless")
amazon_reviews_test_beauty_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "beauty")
amazon_reviews_test_drugstore_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "drugstore")
amazon_reviews_test_kitchen_id = DatasetID("amazon_reviews_multi", "en", "test", "product_category", "kitchen")

SUPER_GLUE = [
    boolq_train_id,
    cb_train_id,
    copa_train_id,
    multirc_train_id,
    record_train_id,
    rte_train_id,
    wic_train_id,
    wsc_train_id
]
SUPER_GLUE_TEST = [
    boolq_test_id,
    cb_test_id,
    copa_test_id,
    multirc_test_id,
    record_test_id,
    rte_test_id,
    wic_test_id,
    wsc_test_id
]

XNLI = [
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
]
XNLI_TEST = [
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
]

AMAZON_REVIEWS = [
    amazon_reviews_home_id,
    amazon_reviews_apparel_id,
    amazon_reviews_wireless_id,
    amazon_reviews_beauty_id,
    amazon_reviews_drugstore_id,
    amazon_reviews_kitchen_id,
]
AMAZON_REVIEWS_TEST = [
    amazon_reviews_test_home_id,
    amazon_reviews_test_apparel_id,
    amazon_reviews_test_wireless_id,
    amazon_reviews_test_beauty_id,
    amazon_reviews_test_drugstore_id,
    amazon_reviews_test_kitchen_id,
]