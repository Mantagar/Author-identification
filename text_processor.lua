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
	file=io.open(filename,"r")
	text=file:read("*a")
	file:close()

	--pick char and convert to tensor
	local data = {}
	for i=1,#text do
		data[i]=TP.charToTensor(text:sub(i,i),alphabet)
	end
	
	return data
end

return TP
