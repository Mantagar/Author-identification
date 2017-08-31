local Utils={}
local Colorizer=require 'Colorizer'
local TP=require 'data/text_processor.lua'

function Utils.getKnownData(amount)
	local i=1
	local data={}
	local pfile = io.popen('ls data/known | head -n '..amount)
	for filename in pfile:lines() do
		print("  Author "..i.."\t"..Colorizer.green("\u{2713}"))
		data[i]=torch.load('./data/known/'..filename)
		i = i + 1
	end
	pfile:close()
	return data
end

function Utils.getUnknownData(amount)
	local i=1
	local data={}
	local pfile = io.popen('ls data/unknown | head -n '..amount)
	for filename in pfile:lines() do
		print("  Unknown "..i.."\t"..Colorizer.green("\u{2713}"))
		data[i]=torch.load('./data/unknown/'..filename)
		i = i + 1
	end
	pfile:close()
	return data
end

function Utils.loadAnswers(filename,amount)
	local file=io.open(filename,"r")
	local text=file:read("*a")
	file:close()
	local words={}
	for word in string.gmatch(text,"[^%s]+") do
		table.insert(words,word)
	end
	local mapping={}
	mapping["Y"]=true
	mapping["N"]=false
	local answers={}
	for i=2,#words,2 do
		table.insert(answers,mapping[words[i]])
		if #answers==amount then
			break
		end
	end
	return answers
end

function Utils.getMatchRate(rnn,data,alphabet)
	local matched=torch.zeros(rnn.heads);
	--Initialize the initial hidden state with zero matrices
	local hiddenState={}
	for r=1,rnn.rnn_size do
		hiddenState[r]=torch.zeros(rnn.hidden_size)
	end
	for iteration=1,#data-1 do
		input=data[iteration]
		rnn[1]:forward({input,table.unpack(hiddenState)})
		--Initialize next hidden state
		for r=1,rnn.rnn_size do
			hiddenState[r]=rnn[1].output[r+rnn.heads]:clone()
		end
		if iteration>10 then 
			--Initialize target
			local target=data[iteration+1]
			for h=1,rnn.heads do
				if TP.tensorToChar(target,alphabet)==TP.tensorToChar(rnn[1].output[h],alphabet) then matched[h]=matched[h]+1 end
			end
		end
	end
	return matched/(#data-11)
end


return Utils
