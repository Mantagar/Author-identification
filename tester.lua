torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'


function generateText(rnn,alphabet,length,seed,head)
	local text=""
	local data = {}
	for i=1,rnn.seq_size do
		data[i]=TP.charToTensor(seed:sub(i%#seed+1,i%#seed+1),alphabet)
	end
	for i=1,length do
		--Initialize the initial hidden state with zero matrices
		local hiddenState={}
		hiddenState[0]={}
		for r=1,rnn.rnn_size do
			hiddenState[0][r]=torch.zeros(rnn.hidden_size)
		end
		--Forward through time
		for t=1,rnn.seq_size do
			--Pass input and previous hidden state at timestep t
			rnn[t]:forward({data[t],table.unpack(hiddenState[t-1])})
			--Initialize next hidden state
			hiddenState[t]={}
			for r=1,rnn.rnn_size do
				hiddenState[t][r]=rnn[t].output[r+rnn.heads]:clone()
			end
		end
		local char=TP.tensorToChar(rnn[rnn.seq_size].output[head],alphabet)
		text=text..char
		table.remove(data,1)
		data[rnn.seq_size]=TP.charToTensor(char,alphabet)
	end
	return text
end


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

local rnn=RNN.loadSnapshot("1496759159.rnn")

local data1=TP.convertFileIntoData('data/author1',TP.enAlphabet)
local data2=TP.convertFileIntoData('data/author2',TP.enAlphabet)
local data3=TP.convertFileIntoData('data/author3',TP.enAlphabet)
local unknown=TP.convertFileIntoData('data/unknown',TP.enAlphabet)



print("--------Text generated for head 1------------")
print(generateText(rnn,TP.enAlphabet,100,"someseed",1))
print("--------Text generated for head 2------------")
print(generateText(rnn,TP.enAlphabet,100,"someseed",2))
print("--------Text generated for head 3------------")
print(generateText(rnn,TP.enAlphabet,100,"someseed",3))


print("Head 1 matches its data at rate: ",getMatchRate(rnn,data1,1))
print("Head 2 matches its data at rate: ",getMatchRate(rnn,data2,2))
print("Head 3 matches its data at rate: ",getMatchRate(rnn,data3,3))

print("Head 1 matches unknown at rate[should win]: ",getMatchRate(rnn,unknown,1))
print("Head 2 matches unknown at rate: ",getMatchRate(rnn,unknown,2))
print("Head 3 matches unknown at rate: ",getMatchRate(rnn,unknown,3))
