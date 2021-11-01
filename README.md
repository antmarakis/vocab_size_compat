### Bible

The data here is the usual PBC data, with the metadata information removed (and stored in `*-x-bible-newworld2013.train` files).

To run the experiments, use the following line:

`source run_bible.sh ARGS`

For `ARGS`, see the `run_bible.sh` script.


### Wikipedia

Data is formatted line-by-line. The way I have it, each line is a single sentence (split on punctuation). An example of such data (for pretraining) is in /mounts/data/proj/antmarakis/wikipedia/wikipedia_en_el_seq.txt where English and Greek data are concatenated. For the tokenization, separate files are needed for the two languages, again formatted line-by-line.

To run the experiments, use the following line:

`source run_wikipedia.sh EN_SIZE OTHER_SIZE OTHER_LANG`

### Static Embeddings

After downloading the [Public Bible Corpus](http://www.lrec-conf.org/proceedings/lrec2014/pdf/220_Paper.pdf), run the bash script `run.sh` (you will need to update the `WORKDIR` and `BIBLEDIR` variables) found in the `static` folder.
