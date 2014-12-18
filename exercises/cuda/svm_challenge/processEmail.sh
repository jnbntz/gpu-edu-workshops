#!/bin/bash

#
#  Copyright 2014 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

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
