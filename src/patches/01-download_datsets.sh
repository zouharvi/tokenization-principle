#!/usr/bin/bash

mkdir -p data
cd data

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