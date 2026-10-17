#!/usr/bin/bash

mkdir -p data
cd data

wget https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/cs-en.txt.zip
unzip cs-en.txt.zip

mkdir -p cs-en
head -n 11000000 CCAligned.cs-en.en > cs-en/orig.en
head -n 11000000 CCAligned.cs-en.cs > cs-en/orig.cs

# create split
head -n 10000000 cs-en/orig.en > cs-en/train.en
head -n 10000000 cs-en/orig.cs > cs-en/train.cs
tail -n 1000000  cs-en/orig.en | head -n 500000 > cs-en/dev.en
tail -n 1000000  cs-en/orig.cs | head -n 500000 > cs-en/dev.cs
tail -n 500000 cs-en/orig.en > cs-en/test.en
tail -n 500000 cs-en/orig.cs > cs-en/test.cs

rm CCAligned.cs-en.xml README LICENSE cs-en.txt.zip CCAligned.cs-en.{en,cs}





wget https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/en-zh.txt.zip
unzip en-zh.txt.zip

mkdir -p zh-en
head -n 11000000 ParaCrawl.en-zh.en > zh-en/orig.en
head -n 11000000 ParaCrawl.en-zh.zh > zh-en/orig.zh

# create split
head -n 10000000 zh-en/orig.en > zh-en/train.en
head -n 10000000 zh-en/orig.zh > zh-en/train.zh
tail -n 1000000  zh-en/orig.en | head -n 500000 > zh-en/dev.en
tail -n 1000000  zh-en/orig.zh | head -n 500000 > zh-en/dev.zh
tail -n 500000 zh-en/orig.en > zh-en/test.en
tail -n 500000 zh-en/orig.zh > zh-en/test.zh

rm ParaCrawl.en-zh.xml README LICENSE en-zh.txt.zip ParaCrawl.en-zh.{en,zh}



wget https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/de-en.txt.zip
unzip de-en.txt.zip

mkdir -p de-en
head -n 11000000 CCAligned.de-en.en > de-en/orig.en
head -n 11000000 CCAligned.de-en.de > de-en/orig.de

# create split
head -n 10000000 de-en/orig.en > de-en/train.en
head -n 10000000 de-en/orig.de > de-en/train.de
tail -n 1000000  de-en/orig.en | head -n 500000 > de-en/dev.en
tail -n 1000000  de-en/orig.de | head -n 500000 > de-en/dev.de
tail -n 500000 de-en/orig.en > de-en/test.en
tail -n 500000 de-en/orig.de > de-en/test.de

rm CCAligned.de-en.xml README LICENSE de-en.txt.zip CCAligned.de-en.{en,de}


cd ..