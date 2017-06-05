torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'text_processor.lua'

local data1=TP.convertFileIntoData('data/author1',TP.enAlphabet)
local data2=TP.convertFileIntoData('data/author2',TP.enAlphabet)
local data3=TP.convertFileIntoData('data/author3',TP.enAlphabet)


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



require 'nn'
require 'nngraph'
local RNN=require 'RNN'

local isize=#TP.enAlphabet
local hsize=256
local rnn_size=2
local seq_size=10
local heads=3
local model=RNN.createModel(isize,hsize,rnn_size,heads)
graph.dot(model.fg,"multihead-model","mulithead-model")

local rnn=RNN.unfoldModel(model,seq_size)

for i=1,100 do
RNN.trainUnfoldedModel(rnn,0.0002,data1,1)
RNN.trainUnfoldedModel(rnn,0.0002,data2,2)
RNN.trainUnfoldedModel(rnn,0.0002,data3,3)
print("1-----"..generateText(rnn,TP.enAlphabet,100,"seed",1))
print("2-----"..generateText(rnn,TP.enAlphabet,100,"seed",2))
print("3-----"..generateText(rnn,TP.enAlphabet,100,"seed",3))
end

