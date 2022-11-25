#!/usr/bin/bash

cd data


wget https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/de-en.txt.zip
unzip de-en.txt.zip

mkdir -p CCrawl.de-en
head -n 10000000 CCAligned.de-en.en > CCrawl.de-en/orig.en
head -n 10000000 CCAligned.de-en.de > CCrawl.de-en/orig.de

rm CCAligned.de-en.xml README LICENSE de-en.txt.zip CCAligned.de-en.{en,de}

cd ..