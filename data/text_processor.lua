local TP={}

TP.enAlphabet={
' ',
'e',
't',
'o',
'a',
'n',
'i',
's',
'h',
'r',--10
'\n',
'l',
'd',
'u',
'y',
'm',
'g',
'w',
',',
"'",--20
'.',
'f',
'c',
'b',
'p',
'-',
'k',
'¹',--used as capital letter prefix
'v',
'!',--30
'?',
';',
'j',
'q',
'x',
'"',
'z',
':',
'(',
')'--40
}
TP.esAlphabet={
' ',
'e',
'a',
'o',
's',
'n',
'i',
'r',
'l',
'd',--10
'c',
't',
'u',
'm',
'p',
'\u{0301}',
',',
'b',
'g',
'q',--20
'v',
'.',
'y',
'f',
'h',
'z',
'j',
'\n',
'¹',--used as capital letter prefix
'x',--30
'\u{0303}',
'"',
'7',
':',
'\u{2014}',
'¿',
'?',
'(',
')',
';',--40
'k',
'-',
'/',
'w',
"'",
'%',
'!',
'«',
'»',
'¡',--50
'@',
'\u{0308}'
}
TP.grAlphabet={
' ',
"α",
'\u{0301}',
"ο",
"ι",
"ε",
"τ",
"ν",
"υ",
"η",--10
"ρ",
"σ",
"π",
"κ",
"μ",
"ς",
"λ",
"ω",
"γ",
"δ",--20
'.',
',',
"χ",
"θ",
"φ",
"β",
"ζ",
"ξ",
'\n',
"¹",--30
"«",
"»",
"ψ",
'7',
';',
's',--latin replacement
'-',
'!',
')',
'(',--40
':',
'"',
'\u{0308}',
"'",
'`',
'%',
'  ',
'/',
'@',
}
TP.nlAlphabet={
' ',
'e',
'n',
'a',
'i',
't',
'o',
'r',
'd',
'l',--10
's',
'g',
'k',
'v',
'm',
'h',
'u',
'j',
'w',
'b',--20
'p',
'z',
'c',
'.',
'f',
',',
"¹",
'y',
"'",
'7',--30
'-',
')',
'(',
'\u{0308}',
'?',
':',
'\n',
"'",
'x',
'\u{0301}',--40
'q',
'"',
'!',
';',
'&',
'%',
'\u{0300}',
'\u{2014}',
'\u{0302}',
}
function TP.charToTensor(char,alphabet)
--returns tensor with of zeros and one one where position of the char is in the alphabet
	local tensor=torch.Tensor(#alphabet)
	for i=1,#alphabet do
		if(char==alphabet[i]) then
			tensor[i]=1
		else
			tensor[i]=0
		end
	end
	return tensor
end

function TP.tensorToChar(tensor,alphabet)
	local _,index=tensor:max(1)
	return alphabet[index[1]]
end

function TP.convertFileIntoData(filename,alphabet)
	local file=io.open(filename,"r")
	local text=file:read("*a")
	file:close()

	--pick utf8 char and convert to tensor
	local data={}
  data.alphabet=alphabet
	local i=1
	for _,c in utf8.codes(text) do
		data[i]=TP.charToTensor(utf8.char(c),alphabet)
		i=i+1
	end
	
	return data
end

return TP
