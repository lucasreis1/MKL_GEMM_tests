#!/bin/bash

FILE=$1
OUTPUT=$2

cat $FILE | awk '{print $5"x"$8"x"$11","$1"," $15","$23}' | sed '1isize,type,gops,execution_time' > $OUTPUT
