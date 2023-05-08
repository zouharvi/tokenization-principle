#!/usr/bin/bash

mkdir -p data
cd data

wget https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/en-zh.txt.zip
unzip en-zh.txt.zip

mkdir -p PCrawl.zh-en
head -n 10000000 ParaCrawl.en-zh.en > PCrawl.zh-en/orig.en
head -n 10000000 ParaCrawl.en-zh.zh > PCrawl.zh-en/orig.zh

# create split
head -n 9000000 PCrawl.zh-en/orig.en > PCrawl.zh-en/train.en
head -n 9000000 PCrawl.zh-en/orig.zh > PCrawl.zh-en/train.zh
tail -n 1000000  PCrawl.zh-en/orig.en | head -n 500000 > PCrawl.zh-en/zhv.en
tail -n 1000000  PCrawl.zh-en/orig.zh | head -n 500000 > PCrawl.zh-en/zhv.zh
tail -n 500000 PCrawl.zh-en/orig.en > PCrawl.zh-en/test.en
tail -n 500000 PCrawl.zh-en/orig.zh > PCrawl.zh-en/test.zh

rm ParaCrawl.en-zh.xml README LICENSE en-zh.txt.zip ParaCrawl.en-zh.{en,zh}

cd ..