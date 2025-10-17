# !/bin/bash
set -eu -o pipefail

cd data

# Preprocessed HoVer data (3 MB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/hover.tar.gz
tar -xvzf hover.tar.gz

# Preprocessed Wikipedia Abstracts 2017 collection (1 GB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.tar.gz
tar -xvzf wiki.abstracts.2017.tar.gz

# Checkpoints for Baleen retrieval and condesening (8 GB compressed)
wget https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/hover.checkpoints-v1.0.tar.gz
tar -xvzf hover.checkpoints-v1.0.tar.gz