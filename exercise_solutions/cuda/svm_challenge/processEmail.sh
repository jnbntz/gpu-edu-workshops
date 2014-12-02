#!/bin/bash

inputEmail="emailSample1.txt"

#cat $inputEmail | awk '{print tolower($0)}'

awk '{print tolower($0)}' $inputEmail | \
awk '{print gensub(/[[:digit:]]+/,"number","g")}' | \
awk '{print gensub(/(http|https)\:\/\/[[:graph:]]*/,"httpaddr","g")}' | \
awk '{print gensub(/[[:graph:]]+@[[:graph:]]+/,"emailaddr","g")}' | \
awk '{print gensub(/[$]+/,"dollar","g")}' | \
awk '{print gensub(/([^[:alnum:]|^[:blank:]])/,"","g")}' | \
awk 'NF > 0'
