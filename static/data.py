import collections
import numpy as np
from typing import Any, Dict, Callable, Text, Tuple, List, Set
import os
from tqdm import tqdm

from utils.utils import rec_dd, get_logger
LOG = get_logger(__name__)


class Corpus(object):
    """docstring for Corpus"""

    def __init__(self) -> None:
        super(Corpus, self).__init__()
        # Format: Language, SubID, Sentence ID, Text
        self.text = rec_dd()

    @staticmethod
    def apply_function(corpusview: Dict,
                       modification: Callable[[Any], Any]) -> Dict:
        new_view = rec_dd()
        for lang in corpusview:
            for subid in corpusview[lang]:
                for sentenceid in corpusview[lang][subid]:
                    new_view[lang][subid][sentenceid] = modification(
                        corpusview[lang][subid][sentenceid])
        return new_view

    def tokenize(self, tokenize_function: Callable[[Text], List[Text]]) -> None:
        self.tokenized_text = self.apply_function(self.text, tokenize_function)

    def get_ids(self, id_function: Callable[[List[Text]], List[int]]) -> None:
        self.ids = self.apply_function(self.tokenized_text, id_function)

    def get_all_sentence_ids(self) -> List[Any]:
        all_ids = []
        for lang in self.text:
            for subid in self.text[lang]:
                all_ids.extend(list(self.text[lang][subid].keys()))
        return all_ids

    def keep_first_n_sentences(self, n):
        views = [self.text, self.tokenized_text, self.ids]
        for view in views:
            for lang in view:
                for subid in view[lang]:
                    for i, sentenceid in enumerate(list(view[lang][subid].keys())):
                        if i >= n:
                            del view[lang][subid][sentenceid]

    def keep_sentences(self, sentenceids):
        sentenceids = set(sentenceids)
        views = [self.text, self.tokenized_text, self.ids]
        for view in views:
            for lang in view:
                for subid in view[lang]:
                    for i, sentenceid in enumerate(list(view[lang][subid].keys())):
                        if sentenceid not in sentenceids:
                            del view[lang][subid][sentenceid]


class PBC(Corpus):
    """docstring for PBC"""

    def __init__(self, path: Text) -> None:
        super(PBC, self).__init__()
        self.path = path
        self.partition = None
        self.rel_verses = set()
        self.editions = set()

    def get_datasplit(self, path: Text, partition: Text) -> None:
        self.partition = partition
        with open(path, "r") as fp:
            for line in fp:
                part, verse_id = line.strip().split()
                if part == partition:
                    self.rel_verses.add(verse_id)
        LOG.info("Loaded {} verse ids of part {}.".format(len(self.rel_verses), partition))

    def get_relevant_editions(self, path: Text) -> None:
        editions = set()
        with open(path, "r") as fp:
            for line in fp:
                edition = line.strip().split()[0]
                editions.add(edition)
        return editions

    def convert_edition_to_filename(self, edition: Text) -> Text:
        l, t = edition.split("_")[0], '_'.join(edition.split("_")[1:])
        if t != '':
            t = '-' + t
        t = t.replace("_", "-")
        filename = self.path + l + "-x-bible" + t + ".txt"
        return filename

    def convert_filename_to_edition(self, filename: Text) -> Text:
        filename = filename.replace(self.path, '')
        filename = filename.replace('.txt', '')
        filename = filename.replace('-x-bible', '_')
        if filename[-1] == '_':
            filename = filename[:-1]
        filename = filename.replace('-', '', 1)
        return filename

    def load_single_edition(self, edition: Text) -> Dict[Text, Text]:
        filename = self.convert_edition_to_filename(edition)
        fin = open(filename, 'r')
        data = collections.defaultdict(str)
        for line in fin:
            line = line.replace("\n", "")
            if line[0] == '#':
                continue
            if len(line.split('\t')) != 2:
                continue
            verse_id, verse = line.split('\t')
            # check if verse is empty
            if not verse.strip():
                continue
            # skip old testament
            if int(verse_id[0]) < 4:
                continue
            if self.partition is None or verse_id in self.rel_verses:
                data[verse_id] = verse
        return data

    def load_editions(self, editions: List[Text]) -> None:
        for edition in tqdm(editions):
            self.editions.add(edition)
            language = edition.split("_")[0]
            self.text[language][edition] = self.load_single_edition(edition)

    def get_all_edition_names(self) -> List[Text]:
        filenames = os.listdir(self.path)
        filenames = [x for x in filenames if "-bible" in x]
        all_editions = [self.convert_filename_to_edition(x) for x in filenames]
        return all_editions
