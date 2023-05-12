import argparse
from tqdm import tqdm
import os
import ipdb
import numpy
from collections import defaultdict
import pathlib

def add_args(parser):
    parser.add_argument('--train', default="../data/mono_en/train.txt")
    parser.add_argument('--val', default="../data/mono_en/valid.txt")
    parser.add_argument('--test', default="../data/mono_en/test.txt")
    parser.add_argument('--out_dir', default="./data/mono_en")
    args = parser.parse_args()
    return args

def triples2ids(triples, ent2id, rel2id):
    triplesInids = []
    for (s,r,o) in triples:
        triplesInids.append((ent2id[s], rel2id[r], ent2id[o]))
    return triplesInids

def get_trip_lines(triples):
        trip_lines = [] # [str(len(triples))]
        for (s,r,o) in triples:
            trip_lines.append(str(s) + '\t' + str(r) + '\t' + str(o) + '\n')
        return trip_lines

def transform(train_path, val_path, test_path, out_dir):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_triples = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_triples = f.readlines()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_triples = f.readlines()

    train_triples = [triple.strip().split('\t') for triple in train_triples]
    val_triples = [triple.strip().split('\t') for triple in val_triples]
    test_triples = [triple.strip().split('\t') for triple in test_triples]

    print("\nLoaded %u train, %u val, %u test triples"%(len(train_triples), len(val_triples), len(test_triples)))

    subjects, relations, objects = [], [], []
    for (s,r,o) in train_triples:
        s, r, o = str(s), str(r), str(o)
        subjects.append(s)
        relations.append(r)
        objects.append(o)
    for (s,r,o) in val_triples:
        s, r, o = str(s), str(r), str(o)
        subjects.append(s)
        relations.append(r)
        objects.append(o)
    for (s,r,o) in test_triples:
        s, r, o = str(s), str(r), str(o)
        subjects.append(s)
        relations.append(r)
        objects.append(o)

    NPs = list(set(subjects + objects))
    RPs = list(set(relations))

    rel2id, id2rel, rel2id_lines = {}, {}, []
    # rel2id_lines.append(str(len(RPs)))
    for rel_id, rel in enumerate(RPs):
        assert not (rel in rel2id.keys() and rel_id in id2rel.keys()), ipdb.set_trace()
        rel2id[rel] = rel_id
        id2rel[rel_id] = rel
        # rel2id_lines.append(rel + "\t" + str(rel_id))
        rel2id_lines.append(str(rel_id) + '\t' + rel + '\n')

    ent2id, id2ent, ent2id_lines = {}, {}, []
    # ent2id_lines.append(str(len(NPs)))
    ent2text_lines = []
    for ent_id, ent in enumerate(NPs):
        assert not (ent in ent2id.keys() and ent_id in id2ent.keys()), ipdb.set_trace()
        ent2id[ent] = ent_id
        id2ent[ent_id] = ent
        # ent2id_lines.append(ent + "\t" + str(ent_id))
        ent2id_lines.append(str(ent_id) + '\t' + ent + '\n')
        
        ent2text_lines.append(str(ent_id) + '\t' + ent + '\n')

    with open(os.path.join(out_dir, 'relation.txt'), 'w', encoding='utf-8') as rel2id_file:
        rel2id_file.writelines(''.join(rel2id_lines))
    with open(os.path .join(out_dir, 'entity.txt'), 'w', encoding='utf-8') as ent2id_file:
        ent2id_file.writelines(''.join(ent2id_lines))
    with open(os.path .join(out_dir, 'text.txt'), 'w', encoding='utf-8') as ent2text_file:
        ent2text_file.writelines(''.join(ent2text_lines))
    print("\nSaved relation.txt, entity.txt, and text.txt in %s"%(out_dir))
    
    train_trip = triples2ids(train_triples, ent2id, rel2id)
    valid_trip = triples2ids(val_triples, ent2id, rel2id)
    test_trip = triples2ids(test_triples, ent2id, rel2id)
    
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as trip_file:
        trip_file.writelines(''.join(get_trip_lines(train_trip)))
    with open(os.path.join(out_dir, 'valid.txt'), 'w', encoding='utf-8') as trip_file:
        trip_file.writelines(''.join(get_trip_lines(valid_trip)))
    with open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8') as trip_file:
        trip_file.writelines(''.join(get_trip_lines(test_trip)))
    print("\nSaved train.txt, valid.txt and test.txt in %s"%(out_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    train_path = args.train
    val_path = args.val
    test_path = args.test
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    transform(train_path, val_path, test_path, out_dir)