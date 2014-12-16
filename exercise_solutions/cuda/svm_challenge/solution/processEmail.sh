#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "illegal number of arguments"
  echo "Usage: $0 <filename>"
  exit
fi

if [ ! -f "$1" ]; then
  echo "$1 is not a file"
  exit
fi

inputEmail=$1

#cat $inputEmail | awk '{print tolower($0)}'

awk '{print tolower($0)}' $inputEmail | \
awk '{print gensub(/[[:digit:]]+/,"number","g")}' | \
awk '{print gensub(/(http|https)\:\/\/[[:graph:]]*/,"httpaddr","g")}' | \
awk '{print gensub(/[[:graph:]]+@[[:graph:]]+/,"emailaddr","g")}' | \
awk '{print gensub(/[$]+/,"dollar","g")}' | \
awk '{print gensub(/([^[:alnum:]|^[:blank:]])/,"","g")}' | \
awk 'NF > 0' > qwerty.txt

./x.porterStemmer qwerty.txt

python processEmail.py > emailVector.txt

rm -f qwerty.txt
