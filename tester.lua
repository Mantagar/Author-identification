torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'data/text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'


function getMatchRate(rnn,data,head)
	local matched=0;
	local tests=#data-rnn.seq_size
	for iteration=1,tests do
		--Initialize the initial hidden state with zero matrices
		local hiddenState={}
		hiddenState[0]={}
		for r=1,rnn.rnn_size do
			hiddenState[0][r]=torch.zeros(rnn.hidden_size)
		end
		--Initialize inputs at consecutive timesteps
		local input={}
		for t=1,rnn.seq_size do
			input[t]=data[iteration+t-1]
		end
		--Forward through time
		for t=1,rnn.seq_size do
			--Pass input and previous hidden state at timestep t
			rnn[t]:forward({input[t],table.unpack(hiddenState[t-1])})
			--Initialize next hidden state
			hiddenState[t]={}
			for r=1,rnn.rnn_size do
				hiddenState[t][r]=rnn[t].output[r+rnn.heads]:clone()
			end
		end
		--Initialize target
		local target=data[iteration+rnn.seq_size]
		if TP.tensorToChar(target,TP.enAlphabet)==TP.tensorToChar(rnn[rnn.seq_size].output[head],TP.enAlphabet) then matched=matched+1 end
	end
	return matched/tests
end

local rnn=RNN.loadSnapshot("1496923741.rnn")


function getAuthorsData()
	local i=1
	local authors={}
	local pfile = io.popen('ls data/binary_authors')
	for filename in pfile:lines() do
		authors[i]=torch.load('./data/binary_authors/'..filename)
		i = i + 1
	end
	pfile:close()
	return authors
end
print("Loading authors' data...")
local authors=getAuthorsData()
print("Loaded "..#authors.." authors")

local unknown=TP.convertFileIntoData("valid1.txt",TP.enAlphabet)
for i=1,rnn.heads do
	--print("Match rate for head "..i.." = "..getMatchRate(rnn,unknown,i))
end

for i=1,#authors do
	for j=1,#authors[i] do
		print("Match rate for head "..i.." document "..j.." = "..getMatchRate(rnn,authors[i][j],i))
	end
end
