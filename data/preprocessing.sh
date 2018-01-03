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
DIRNAME=`dirname $0`
cd "$DIRNAME"
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
reduced="./reduced_authors"
known="./known"
unknown="./unknown"
map="./mappings/map.py"

echo -e "\e[32;2;1mCleaning workplace directories\e[0m"
rm -R $reduced
rm -R $known
rm -R $unknown
mkdir $reduced
mkdir $known
mkdir $unknown

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
	known_file="$known/$author.t7"
	unknown_file="$unknown/$author.t7"
	echo "  $reduced/$author/*"
	for document in "$reduced/$author/*"
	do
		th serializer.lua $known_file $unknown_file $language $document
	done
done
