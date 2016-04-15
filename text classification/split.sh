#!/bin/bash

for i in $(seq 0 4)
do
	mkdir $i
done

for label in `ls -d */`
do
	echo $label
	cd $label
	cnt=0
	for file in `ls`
	do
		folder=$[$cnt/40]
		if [ ! -d ../$folder/$label ]
		then
			mkdir ../$folder/$label
		fi
		cp $file ../$folder/$label/$file
		((cnt++))
	done
	cd ..
done
