#!/bin/bash
#This script converts files in ./authors/ into files with reduced alphabet in ./reduced_authors/ and then to torch serialized data of tensors in ./binary_authors/
# It expects one argument - language of the texts used, valid options are: -en (English), -es (Spanish), -gr (Greek), -nl (Dutch)
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
if [ $# -ne 1 ]; then
	echo -e "\e[31;1mAborting! Expecting exactly one argument.\e[0m"
	exit
fi
case $1 in
	-en) language="en" ;;
	-es) language="es" ;;
	-gr) language="gr" ;;
	-nl) language="nl" ;;
	*)
		echo -e "\e[31;1mAborting! Not a valid argument.\e[0m"
		exit
esac
original="./authors"
binary="./binary_authors"
reduced="./reduced_authors"
map="./mappings/map.py"

echo -e "\e[32;2;1mCleaning workplace directories\e[0m"
rm -R $binary
rm -R $reduced
mkdir $binary
mkdir $reduced

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

for author in `ls $reduced`
do
	torch_file="$binary/$author.t7"
	echo "Converting $reduced/$author/* ---> $torch_file"
	for document in "$reduced/$author/*"
	do
		th serializer.lua $torch_file $language $document
	done
done
