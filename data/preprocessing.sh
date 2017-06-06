#!/bin/bash
#This script converts files in ./authors/ into files with reduced alphabet in ./reduced_authors/ and then to torch serialized data of tensors in ./binary_authors/
#For now it can only process english files TODO expand in the future
#Example structure of ./authors/ directory
#
#authors/
#		William Shakespeare/
#				Hamlet.txt
#				Macbeth.txt
#		Adam Mickiewicz/
#				Reduta Ordona.txt
#
#
language="en"
original="./authors"
binary="./binary_authors"
reduced="./reduced_authors"
map="./mappings/map.py"

echo -e "\e[32;2;1mCleaning workplace directories\e[0m"
rm -R $binary/*
rm -R $reduced/*

echo -e "\e[32;2;1mMapping files\e[0m"
for author in `ls $original`
do
	echo $author
	mkdir $reduced/$author
	for document in `ls $original/$author`
	do
		echo "	"$document
		python $map $language $original/$author/$document > $reduced/$author/$document
	done
done

echo -e "\e[32;2;1mConverting mapped files to tensor data\e[0m"
counter=1
for author in $reduced/*
do
	torch_file="$binary/author$counter"
	echo "Converting $author/* ---> $torch_file"
	for document in "$author/*"
	do
		th serializer.lua $torch_file $document
	done
	let counter=counter+1
done
