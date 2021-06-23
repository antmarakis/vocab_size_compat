from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
from random import shuffle
import numpy as np
import codecs, sys, common_lines, torch
torch.manual_seed(0)
np.random.seed(0)

eval_data = sys.argv[3]
lang = eval_data[:3]
max_len = 256
if lang == 'id-': lang = 'id'
if len(sys.argv) == 6: max_len = int(sys.argv[5])
epochs = 50

word_embedding_model = models.Transformer('/mounts/data/proj/antmarakis/bible/lms/{}'.format(sys.argv[4].replace(' ', '')), max_seq_length=max_len)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# TRAIN
f = open('en_text_similarity_same.txt')
l = [p.strip() for p in f.readlines()]
f.close()

f = open('en_text_similarity_diff.txt')
l += [p.strip() for p in f.readlines()]
f.close()

f = open('{}_text_similarity_same.txt'.format(lang))
l += [p.strip() for p in f.readlines()]
f.close()

f = open('{}_text_similarity_diff.txt'.format(lang))
l += [p.strip() for p in f.readlines()]
f.close()

shuffle(l)
l_train = l

texts_train = [p[:-2].split('\t') for p in l_train]
labels_train = [float(p[-1]) for p in l_train]

train_examples = []
for t, l in zip(texts_train, labels_train):
    train_examples.append(InputExample(texts=t, label=l))

train_loss = losses.CosineSimilarityLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)


def compute_hits(mode='dev'):
    common_lines.parse("/mounts/data/proj/antmarakis/bible/eng-x-bible-newworld2013.{}".format(mode), '/mounts/data/proj/antmarakis/bible/{}.{}'.format(eval_data, mode))

    f = codecs.open("/mounts/data/proj/antmarakis/bible/eng-x-bible-newworld2013_new.{}".format(mode), encoding='utf8')
    en = [l.strip().split('\t')[1] for l in f.readlines()]
    f.close()

    f = codecs.open('/mounts/data/proj/antmarakis/bible/{}_new.{}'.format(eval_data, mode), encoding='utf8')
    el = [l.strip().split('\t')[1] for l in f.readlines()]
    f.close()

    en_embds = model.encode(en, convert_to_tensor=True)
    el_embds = model.encode(el, convert_to_tensor=True)

    # straight
    hits = util.semantic_search(en_embds, el_embds, top_k=10, query_chunk_size=128, corpus_chunk_size=128)

    correct, total = 0, 0
    for i, h in enumerate(hits):
        if i in [s['corpus_id'] for s in h]: correct += 1
        total += 1

    print('en-{}'.format(lang), correct/total)

    f = open('res_sem_both_{}_{}_{}_en_{}.txt'.format(mode, sys.argv[1], sys.argv[2], lang), 'a+')
    f.write(str(correct/total))
    f.write('\n')
    f.close()

    # reverse
    hits = util.semantic_search(el_embds, en_embds, top_k=10, query_chunk_size=128, corpus_chunk_size=128)

    correct, total = 0, 0
    for i, h in enumerate(hits):
        if i in [s['corpus_id'] for s in h]: correct += 1
        total += 1

    print('{}-en'.format(lang), correct/total)

    f = open('res_sem_both_{}_{}_{}_{}_en.txt'.format(mode, sys.argv[1], sys.argv[2], lang), 'a+')
    f.write(str(correct/total))
    f.write('\n')
    f.close()


compute_hits('dev')
compute_hits('test')