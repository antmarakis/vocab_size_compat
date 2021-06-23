import sys
fname = sys.argv[1]

import codecs
f = codecs.open(fname, encoding='utf8')
l = [l.strip() for l in f.readlines()]
f.close()

l = [line.split('\t')[-1] for line in l]

#f = codecs.open('corpus'.join(fname.rsplit('new', 1)), 'w', encoding='utf8')
f = codecs.open('{}_corpus.{}'.format(fname.split('.')[0], fname.split('.')[1]), 'w', encoding='utf8')
f.write('\n'.join(l))
f.close()