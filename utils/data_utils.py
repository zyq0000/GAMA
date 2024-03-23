import copy
import json
import os
import random
from copy import deepcopy
from random import choice, sample

IMG_SHAPE = {
    "vit-base": (577, 768)
}


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def write_json(file_name: str, dataset):
    with open(file_name, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        f.close()


def write_pred_data(args, text_path, predictions):
    result_filename = args.ckpt_path.split("/")[-1].replace(".ckpt", ".json")
    dir_name = os.path.join("./output", args.dataset + "_" + args.task, args.vision_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    test_set = load_data(args, text_path)[-1]
    if args.stage == 1:
        results = {}
        for data, pred in zip(test_set, predictions):
            results[data["id"]] = {
                "id": data["id"],
                "image_path": data["image_path"],
                "pred_1": pred
            }
    elif args.stage == 2:
        results = []
        for data, pred in zip(test_set, predictions):
            data["pred_2"] = pred
            results.append(data)

    write_json(os.path.join(dir_name, result_filename), results)


def remove_duplicated_data(src_test_set):
    src_test_set_no_duplicated = []
    id_set = set()
    for src_data in src_test_set:
        if src_data["id"] in id_set:
            continue
        else:
            id_set.add(src_data["id"])
            src_test_set_no_duplicated.append(src_data)
    return src_test_set_no_duplicated


def random_choice_id(choice_list, exclude_id):
    while True:
        sample_id = choice(choice_list)
        if sample_id != exclude_id:
            return sample_id


def random_sample_id(choice_list, num, exclude_id):
    while True:
        sample_ids = sample(choice_list, num)
        if exclude_id not in sample_ids:
            return sample_ids


def random_sample(choice_list, exclude_id):
    while True:
        chosen_sample = choice(choice_list)
        if exclude_id != chosen_sample["id"]:
            return chosen_sample


def format_neg_sample(query, pos_sample):
    neg_sample = deepcopy(pos_sample)
    neg_sample["2"]["caption"] = query
    neg_sample["2"]["answer"] = "no"
    return neg_sample


def generate_negative_samples_search(train_set):
    processed_dataset = []
    for data in train_set:
        processed_dataset.append(data)
        rand = random.randint(0, 1)
        if rand == 0:
            continue
        negative_sample = random_sample(train_set, data["id"])
        processed_dataset.append(format_neg_sample(data["2"]["caption"], negative_sample))
    return processed_dataset


def pred_as_context(train_set, pred_narratives):
    processed_dataset = []
    for data in train_set:
        new_data = copy.deepcopy(data)
        new_data["2"]["context"] = pred_narratives[data["id"]]["pred_1"] if "id" in data else pred_narratives[data["img_id"]]["pred_1"]
        processed_dataset.append(new_data)
    return processed_dataset


def load_data(args, text_path):
    if args.do_test and args.stage == 1:
        test_set = read_json(os.path.join(text_path, "all.json"))
        print(test_set[0])
        print(test_set[1])
        return [], [], test_set
    if args.do_train:
        train_set = read_json(os.path.join(text_path, "train.json")) + read_json(os.path.join(text_path, "val.json"))
        val_set = read_json(os.path.join(text_path, "test.json"))
    else:
        train_set = []
        val_set = []
    if args.do_test:
        test_set = read_json(os.path.join(text_path, "test.json"))
        print(len(test_set))
    else:
        test_set = []
    if args.stage == 2:
        pred_narratives = read_json(os.path.join("./dataset/localized_narratives_pred", args.dataset + ".json"))
        train_set = pred_as_context(train_set, pred_narratives)
        val_set = pred_as_context(val_set, pred_narratives)
        test_set = pred_as_context(test_set, pred_narratives)
    if args.task == "search" and args.do_train:
        train_set = generate_negative_samples_search(train_set)

    if args.do_train:
        print(train_set[0])
        print(train_set[1])
        print(val_set[0])
    elif args.do_test:
        print(test_set[0])
        print(test_set[1])

    print("train", len(train_set), "val", len(val_set))

    return train_set, val_set, test_set
