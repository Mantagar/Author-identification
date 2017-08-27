torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'data/text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'


function getMatchRate(rnn,data)
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
				if TP.tensorToChar(target,TP.enAlphabet)==TP.tensorToChar(rnn[1].output[h],TP.enAlphabet) then matched[h]=matched[h]+1 end
			end
		end
	end
	return matched/(#data-11)
end

local rnn=RNN.loadSnapshot("1497116017.rnn")


function getAuthorsData(amount)
	local i=1
	local authors={}
	local pfile = io.popen('ls data/binary_authors | head -n '..amount)
	for filename in pfile:lines() do
		print("Author "..i.."...")
		authors[i]=torch.load('./data/binary_authors/'..filename)
		i = i + 1
	end
	pfile:close()
	return authors
end
print("Loading authors' data...")
local authors=getAuthorsData(20)
print("All loaded")


for i=1,#authors do
	for j=1,#authors[i] do
		print("Match rates for author "..i.." document "..j.." = ")
		print(getMatchRate(rnn,authors[i][j]))
	end
end


local unknown={}
unknown[1]=TP.convertFileIntoData("valid1.txt",TP.enAlphabet)
unknown[2]=TP.convertFileIntoData("valid2.txt",TP.enAlphabet)
unknown[3]=TP.convertFileIntoData("valid3.txt",TP.enAlphabet)
for i=1,#unknown do
	print("Match rate for unknown "..i.." = ")
	print(getMatchRate(rnn,unknown[i]))
end
