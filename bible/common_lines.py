"""
python common_lines.py eng-x-bible-newworld2013.txt zho-x-bible-contemp.txt

python common_lines.py eng-x-bible-newworld2013.txt eng-x-bible-newsimplified.txt
"""

import codecs, argparse

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--limit', default=-1)
args = parser.parse_args()

# for some reason, these indices contain unreadable characters
# bad_indices = [1050008, 1050009, 9015006]
bad_indices = []

# READ DATA
f = codecs.open(args.source, 'r', encoding='utf-8')
e = f.readlines()
f.close

f = codecs.open(args.target, 'r', encoding='utf-8')
g = f.readlines()
f.close


def create_dict(lang):
    """Creats dicts in the form {verse_id: verse_txt}"""
    d = {}
    for s in lang:
        index, text = s.split('\t', 1)
        if int(index) in bad_indices:
             # for some reason there is an unreadable char in here
             continue
        d[int(index)] = text.strip()
    return d

e_dict = create_dict(e)
g_dict = create_dict(g)

common_keys = set(g_dict.keys()) & set(e_dict.keys())
common_keys = list(common_keys)[:int(args.limit)]

def del_keys(d, l, to_keep=True):
    """Delete excess keys"""
    for k in list(d.keys()):
        if to_keep:
            if k not in l:
                del d[k]
        else:
            if k in l:
                del d[k]
    return d

e_dict = del_keys(e_dict, common_keys)
g_dict = del_keys(g_dict, common_keys)

def del_empty(s, t):
    deleted = []
    for k, v in list(s.items()):
        if len(v) == 0:
            deleted.append(k)
            del s[k]
    print(deleted)
    t = del_keys(t, deleted, False)
    return s, t

e_dict, g_dict = del_empty(e_dict, g_dict)
g_dict, e_dict = del_empty(g_dict, e_dict)

def write_files(d, n):
    """Write new files, only containing common_keys"""
    f = codecs.open(n, 'w', encoding='utf-8')
    for k, v in d.items():
        f.write('{}\t{}\n'.format(k, v))
    f.close()

s_name = args.source.split('.')[0].split('/')[-1]
t_name = args.target.split('.')[0].split('/')[-1]
s = args.source.split('.')[-1] # set (train/dev/test)
print('{}_new.{}'.format(s_name, s))
write_files(e_dict, '{}_new.{}'.format(s_name, s))
write_files(g_dict, '{}_new.{}'.format(t_name, s))