local TP = require 'text_processor'

local known_data={}
local unknown_data={}
local known_path=arg[1]
local unknown_path=arg[2]
local language=arg[3]
local alphabet=TP.enAlphabet
if language=="es" then
	alphabet=TP.esAlphabet
elseif language=="gr" then
	alphabet=TP.grAlphabet
elseif language=="nl" then
	alphabet=TP.nlAlphabet
end

local ending="unknown.txt"
local known_index=1
local unknown_index=1
for i=4,#arg do
	local name=arg[i]
	local converted=TP.convertFileIntoData(name,alphabet)
	if string.sub(name,-#ending)==ending then
		unknown_data[unknown_index]=converted
		unknown_index=unknown_index+1
		print("did it")
	else
		known_data[known_index]=converted
		known_index=known_index+1
	end
end
torch.save(known_path,known_data)
torch.save(unknown_path,unknown_data)
