# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer


_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)
            # data就是标注文件中的所有内容，包含数据集中所有文档
            for doc in data["documents"]:
                # doc和page等价，对应每一页文档的内容
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]  # 这一页文档中的信息部分
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []  # 这一页文档所有的非other的语义实体[{"start":, "end":, "label":, },{},...]
                relations = []  # 这一页文档中的语义实体关系列表[(),(),(),()]
                id2label = {}  # 语义实体的标签列表{"0": "question", ...}
                entity_id_to_index_map = {}
                empty_entity = set()   # 空语义实体
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]  # 语义实体的标签列表中标签的顺序和语义实体在文档中的顺序是一样的
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])  # [(),(),(),()]
                    tokenized_inputs = self.tokenizer(  # 把每一个语义实体的文本部分输入分词器
                        line["text"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )  # 分词后的输入，也就是把"姓名："分成："姓"，"名"，"："
                    # tokenized_inputs就是文本序列中的一段（[SEP], T1, T2, T3, [SEP]）
                    # tokenized_inputs = {"inputs_ids":[token_id, ...], "offset_mapping":[mapping, ...]}
                    # e.g {'input_ids': [6, 55205, 6959], 'offset_mapping': [(0, 1), (0, 2), (2, 4)]}

                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    last_box = None

                    # tokenized_inputs = {"inputs_ids":[token_id, ...], "offset_mapping":[mapping, ...]}
                    # zip后为[(token_id, offset),(),()...]
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:  # 这是标志位的token_id
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box

                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]
                    # 原本一个line一个label，但是line中有多个字，所以这里的label是一个list，每个line一个label列表
                    # label列表中存放的是这个line(words)中每个字(word)的标签
                    if line["label"] == "other":
                        label = ["O"] * len(bbox)
                    else:
                        label = [f"I-{line['label'].upper()}"] * len(bbox)
                        label[0] = f"B-{line['label'].upper()}"

                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                    # 如果这个line是other，对应的label就是[O,O,O,...]
                    # 否则就是[B-Q,I-Q,...]或[B-A,I-A]或[I-H,I-H,...]
                    if label[0] != "O":  # entities是文档中所有非other的entity集合，index代表entity在这个列表中的索引值
                        entity_id_to_index_map[line["id"]] = len(entities)  # {entity id: index, ...}
                        entities.append(  # [{"start":, "end":, "label":, },{},...]
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),  # 小写字母转大写
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

                relations = list(set(relations))  # [(),(),(),()]去掉了relations中的重复元素
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                # rel是relations中的一个个tuple，rel[0]和rel[1]分别对饮question和answer的line[id]，也就是entity id
                kvrelations = []  # [{"head":, "tail": }, {}, {}]
                for rel in relations:  # id2label = {"0": "question", ...
                    pair = [id2label[rel[0]], id2label[rel[1]]]  # 把两个entity映射回label类型
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            # kvrelations是所有键值对，也就是q-a组合。[{"head": question_index, "tail": answer_index}, {},...]
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )  # {"head": index, "tail": index}
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []  # entities = [{"start":tokenized_doc_id, "end":, "label":, },{},...]
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(  # relations = [{"", "", "", ""}, {}...]
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],  # head和tail都是在entities中的index值
                            "start_index": get_relation_span(rel)[0],  # get_relation_span(rel) = [min, max]
                            "end_index": get_relation_span(rel)[1],
                            # start_index和end_index都是head和tail也就是question和answer的tokenized_doc_id
                        }
                        for rel in kvrelations  # kvrelations = [{"head": question_index, "tail": answer_index}, {},...]
                    ],
                    key=lambda x: x["head"],
                )

                chunk_size = 512  # chunk size是text token序列的长度
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index : index + chunk_size]
                    entities_in_this_span = []
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)
                    relations_in_this_span = []
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item
