# 
# Scripts for Jindřich & Katharina, March 2024
# 

# get requirements
pip3 install morfessor sacremoses

# clone repo
git clone https://github.com/zouharvi/tokenization-principle.git
cd tokenization-principle

# get some data
wget https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/de.txt.gz -P data/
gunzip data/de.txt.gz

# pretokenize whole words with sacremoses
cat data/de.txt | sacremoses -j 20 -l de tokenize > data/de.tok

# TRAIN & APPLY Morfessor
python3 ./src/wrappers/morfessor_wrap_train.py -i data/de.tok -vo data/model_morfessor.pkl
python3 ./src/wrappers/morfessor_wrap_apply.py -i data/de.tok -m data/model_morfessor.pkl -o data/de.morfessor.tok --number-of-lines 1000

# TRAIN & APPLY LZW
python3 ./src/lempel_ziv_welsch/fit_lzw.py -i data/de.tok -vo data/model_lz.vocab
python3 ./src/lempel_ziv_welsch/apply_lzw.py -i data/de.tok -o data/de.lz.tok -vi data/model_lz.vocab --number-of-lines 1000

# TRAIN & APPLY Stochastic BPE with uniform distribution
python3 ./src/fit_bpe.py --model "random" --vocab-output data/model_bperandom_uniform.vocab --randomness-dist "uniform" --vocab-size 8192
python3 ./src/apply_bpe.py -i data/de.tok -o data/de.bperandom_uniform.tok --vocab-input data/model_bperandom_uniform.vocab --number-of-lines 1000

# TRAIN & APPLY Stochastic BPE with softmax distribution (t=1)
python3 ./src/fit_bpe.py --model "random" --vocab-output data/model_bperandom_softmax1.vocab --randomness-dist "softmax" --vocab-size 8192 --randomness-temp 1
python3 ./src/apply_bpe.py -i data/de.tok -o data/de.bperandom_softmax1.tok --vocab-input data/model_bperandom_softmax1.vocab --number-of-lines 1000

# TRAIN & APPLY Stochastic BPE with softmax distribution (t=0.2)
python3 ./src/fit_bpe.py --model "random" --vocab-output data/model_bperandom_softmax02.vocab --randomness-dist "softmax" --vocab-size 8192 --randomness-temp 0.2 
python3 ./src/apply_bpe.py -i data/de.tok -o data/de.bperandom_softmax02.tok --vocab-input data/model_bperandom_softmax02.vocab --number-of-lines 1000

# TRAIN & APPLY Almost greedy BPE (taking 4th most frequent pair)
python3 ./src/fit_bpe.py --model "greedyalmost" --vocab-output data/model_bpegreedy_almost4.vocab --greedy-n 4 --vocab-size 8192
python3 ./src/apply_bpe.py -i data/de.tok -o data/de.model_bpegreedy_almost4.tok --vocab-input data/model_bpegreedy_almost4.vocab --number-of-lines 1000

# take a look at the results
head -n 1 data/de.*.tok
# it should look like this:

# ==> data/de.bperandom_softmax02.tok <==
# Ge@@ ne@@ h@@ m@@ ig@@ ung des Pro@@ to@@ ko@@ ll@@ s der vor@@ ang@@ eg@@ ang@@ en@@ en Si@@ tz@@ ung : sie@@ he Pro@@ to@@ ko@@ ll
# ==> data/de.bperandom_softmax1.tok <==
# Gene@@ hm@@ ig@@ un@@ g d@@ e@@ s Pr@@ o@@ t@@ o@@ k@@ oll@@ s d@@ er v@@ ora@@ n@@ g@@ e@@ g@@ a@@ n@@ g@@ ene@@ n Si@@ tzun@@ g : s@@ ie@@ h@@ e Pr@@ o@@ t@@ o@@ k@@ oll
# ==> data/de.bperandom_uniform.tok <==
# G@@ e@@ n@@ e@@ hmig@@ u@@ n@@ g d@@ es P@@ r@@ ot@@ o@@ k@@ o@@ l@@ l@@ s d@@ e@@ r v@@ o@@ ran@@ g@@ e@@ g@@ an@@ g@@ e@@ n@@ en Sitzu@@ n@@ g : s@@ ie@@ he P@@ r@@ ot@@ o@@ k@@ o@@ l@@ l
# ==> data/de.lz.tok <==
# Gen@@ ehm@@ igung des Protokolls der vora@@ nge@@ gang@@ enen Sitzung : sieh@@ e Protokoll
# ==> data/de.model_bpegreedy_almost4.tok <==
# Gen@@ e@@ h@@ mi@@ gung des Pro@@ to@@ ko@@ ll@@ s der vor@@ ange@@ gan@@ genen Si@@ tzung : sie@@ he Pro@@ to@@ ko@@ ll
# ==> data/de.morfessor.tok <==
# Genehmigung des Protokoll@@ s der voran@@ gegangen@@ en Sitz@@ ung : si@@ ehe Protokoll