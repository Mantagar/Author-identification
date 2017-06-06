local TP = require 'text_processor'

local data={}
local path=arg[1]
for i=2,#arg do
	data[i-1]=TP.convertFileIntoData(arg[i],TP.enAlphabet)
end
torch.save(path,data)
