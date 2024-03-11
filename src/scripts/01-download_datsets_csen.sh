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

cd ..