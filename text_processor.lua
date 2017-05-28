local TP={}

function TP.getAlphabet(string)
--convert to array
	local array = {}
	for i=1,#string do
		array[i]=string:sub(i,i)
	end
--pick only unique characters
	local hash = {}
	local alphabet = {}
	for _,v in ipairs(array) do
		if (not hash[v]) then
			alphabet[#alphabet+1] = v
			hash[v] = true
		end
	end

	return alphabet,array
end

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
	
	local alphabet,array=TP.getAlphabet(text)

	for i=1,#array do
		array[i]=TP.charToTensor(array[i],alphabet)
	end
	
	array.alphabet=alphabet
	return array
end

return TP
