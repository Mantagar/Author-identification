import en
import es
import gr
import nl
import unicodedata
import re
import sys

def nullmapper(x):
    return x

#XXX assumes NFKD norm
def get_charmap(lang):
    if lang is None:
        return nullmapper

    charmap = globals()[lang].charmap

    def mapper(text):
        text = text.decode('utf8')
        text = unicodedata.normalize('NFKD', text)
        text = u''.join(charmap.get(x, u'') for x in text).encode('utf8')
        return re.sub(r'(\w)\1\1\1\1\1+', r'\1\1\1\1\1', text, flags=re.U)

    return mapper

text=""
f=open(sys.argv[1],"r")
for line in f:
	text+=line
f.close()
print get_charmap("en")(text)
