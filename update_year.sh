#!/bin/bash -x

find ./ -type f -exec sed -i 's/Copyright 2014/Copyright 2015/' {} \;
