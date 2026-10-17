#!/usr/bin/bash

mkdir -p data
cd data

wget https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/cs-en.txt.zip
unzip cs-en.txt.zip

mkdir -p CCrawl.cs-en
head -n 10000000 CCAligned.cs-en.en > CCrawl.cs-en/orig.en
head -n 10000000 CCAligned.cs-en.cs > CCrawl.cs-en/orig.cs

# create split
head -n 9000000 CCrawl.cs-en/orig.en > CCrawl.cs-en/train.en
head -n 9000000 CCrawl.cs-en/orig.cs > CCrawl.cs-en/train.cs
tail -n 1000000  CCrawl.cs-en/orig.en | head -n 500000 > CCrawl.cs-en/dev.en
tail -n 1000000  CCrawl.cs-en/orig.cs | head -n 500000 > CCrawl.cs-en/dev.cs
tail -n 500000 CCrawl.cs-en/orig.en > CCrawl.cs-en/test.en
tail -n 500000 CCrawl.cs-en/orig.cs > CCrawl.cs-en/test.cs

rm CCAligned.cs-en.xml README LICENSE cs-en.txt.zip CCAligned.cs-en.{en,cs}





wget https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/en-zh.txt.zip
unzip en-zh.txt.zip

mkdir -p PCrawl.zh-en
head -n 10000000 ParaCrawl.en-zh.en > PCrawl.zh-en/orig.en
head -n 10000000 ParaCrawl.en-zh.zh > PCrawl.zh-en/orig.zh

# create split
head -n 9000000 PCrawl.zh-en/orig.en > PCrawl.zh-en/train.en
head -n 9000000 PCrawl.zh-en/orig.zh > PCrawl.zh-en/train.zh
tail -n 1000000  PCrawl.zh-en/orig.en | head -n 500000 > PCrawl.zh-en/dev.en
tail -n 1000000  PCrawl.zh-en/orig.zh | head -n 500000 > PCrawl.zh-en/dev.zh
tail -n 500000 PCrawl.zh-en/orig.en > PCrawl.zh-en/test.en
tail -n 500000 PCrawl.zh-en/orig.zh > PCrawl.zh-en/test.zh

rm ParaCrawl.en-zh.xml README LICENSE en-zh.txt.zip ParaCrawl.en-zh.{en,zh}



wget https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/de-en.txt.zip
unzip de-en.txt.zip

mkdir -p CCrawl.de-en
head -n 10000000 CCAligned.de-en.en > CCrawl.de-en/orig.en
head -n 10000000 CCAligned.de-en.de > CCrawl.de-en/orig.de

# create split
head -n 9000000 CCrawl.de-en/orig.en > CCrawl.de-en/train.en
head -n 9000000 CCrawl.de-en/orig.de > CCrawl.de-en/train.de
tail -n 1000000  CCrawl.de-en/orig.en | head -n 500000 > CCrawl.de-en/dev.en
tail -n 1000000  CCrawl.de-en/orig.de | head -n 500000 > CCrawl.de-en/dev.de
tail -n 500000 CCrawl.de-en/orig.en > CCrawl.de-en/test.en
tail -n 500000 CCrawl.de-en/orig.de > CCrawl.de-en/test.de

rm CCAligned.de-en.xml README LICENSE de-en.txt.zip CCAligned.de-en.{en,de}


cd ..