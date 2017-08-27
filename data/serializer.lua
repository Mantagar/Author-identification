local TP = require 'text_processor'

local data={}
local path=arg[1]
local language=arg[2]
local alphabet=TP.enAlphabet
if language=="es" then
	alphabet=TP.esAlphabet
elseif language=="gr" then
	alphabet=TP.grAlphabet
elseif language=="nl" then
	alphabet=TP.nlAlphabet
end

for i=3,#arg do
	data[i-2]=TP.convertFileIntoData(arg[i],alphabet)
end
torch.save(path,data)
