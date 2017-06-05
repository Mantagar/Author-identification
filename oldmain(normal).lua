local TP=require 'text_processor.lua'

local data=TP.convertFileIntoData('trainset.txt')

print("Characters in text:",#data)
print("Unique characters:",#data.alphabet)

require 'nn'
require 'nngraph'

function generateText(rnn,alphabet,length,seed)
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
				hiddenState[t][r]=rnn[t].output[r+1]:clone()
			end
		end
		local char=TP.tensorToChar(rnn[rnn.seq_size].output[1],alphabet)
		text=text..char
		table.remove(data,1)
		data[rnn.seq_size]=TP.charToTensor(char,alphabet)
	end
	return text
end
--[[
****** input_size - usually alphabet size
****** hidden_size - size of every hidden state
****** rnn_size - specifies amount of hidden states and should be greater than 0
******
****** Creates reccurent neural network with input_size inputs and outputs where first input and output are meant for data and another
****** rnn_size for hidden states
]]
function createModel(input_size,hidden_size,rnn_size)
	local input={}
	local hidden={}
	local output={}
	input[1]=nn.Identity()()
	for i=2,rnn_size+1 do
		input[i]=nn.Identity()()
		local i2h = input[i] - nn.Linear(hidden_size,hidden_size)
		local h2h = i==2 and input[1] - nn.Linear(input_size,hidden_size) or hidden[i-2] - nn.Linear(hidden_size,hidden_size)
		hidden[i-1] = {h2h,i2h} - nn.CAddTable() - nn.Tanh()
		output[i] = hidden[i-1]
	end
	output[1]=hidden[rnn_size] - nn.Linear(hidden_size,input_size)-- - nn.LogSoftMax()
	--ANNOTATIONS
	--TODO remove all ipairs calls, they are terriblly slow
	for i,each in ipairs(input) do
			each:annotate{name='HIDDEN STATE[t-1]['..(i-1)..']\n', graphAttributes = {style="filled", fillcolor = 'lightblue'}}
	end
	input[1]:annotate{name='INPUT[t]\n', graphAttributes = {style="filled", fillcolor = 'yellow'}}
	for i,each in ipairs(hidden) do
			each:annotate{name='HIDDEN STATE[t]['..i..']\n', graphAttributes = {style="filled", fillcolor = 'red'}}
	end
	output[1]:annotate{name='OUTPUT[t]\n', graphAttributes = {style="filled", fillcolor = 'yellow'}}
	print("INPUT: "..#input,"HIDDEN: "..#hidden,"OUTPUT: "..#output)

	model=nn.gModule(input,output)
	model.input_size=input_size
	model.hidden_size=hidden_size
	model.rnn_size=rnn_size
	return model
end
--Unfolds model in time for sequences of length seq_size and returns appropriate RNN
function unfoldModel(model,seq_size)
	--Clone models in time
	rnn={}
	for i=1,seq_size do
		rnn[i]=model:clone()
	end
	--Share weight, bias, gradWeight and gradBias
	local sharedParams,sharedGradParams=rnn[1]:parameters()
	for t=2,seq_size do
		local params,gradParams=rnn[t]:parameters()
		for i=1,#params do
			params[i]:set(sharedParams[i])
			gradParams[i]:set(sharedGradParams[i])
		end
	end
	rnn.seq_size=seq_size
	rnn.hidden_size=model.hidden_size
	rnn.input_size=model.input_size
	rnn.rnn_size=model.rnn_size
	return rnn
end
--TRAIN UNFOLDED MODEL
function trainUnfoldedModel(rnn,learning_rate,data,epochs)
local avg_err=0;
--Repeat for given amount of epochs
for counter=1,epochs do
	local time=os.clock()
	--Zero initial gradient
	rnn[1]:zeroGradParameters()
	--Iterate over every data element, except the last rnn.seq_size (they are the last targets)
	for iteration=1,#data-rnn.seq_size do
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
			rnn[t]:forward({input[t],table.unpack(hiddenState[t-1])}) --TODO read below
			--We can get the outputs by calling rnn[t].output
			--print(rnn[t].output)
			--Initialize next hidden state
			hiddenState[t]={}
			for r=1,rnn.rnn_size do
				hiddenState[t][r]=rnn[t].output[r+1]:clone()
			end
		end
		--Initialize targets (for ClassNLLCriterion instead of providing array we provide the index of one_hot's 1)
		local target={}
		for t=1,rnn.seq_size do
			_,index=data[iteration+t]:max(1) --TODO remove max call, it is 1.5 times slower
			target[t]=index
		end
		--Calculate error and gragient loss at every timestep
		local err={}
		local gradLoss={}
		for t=1,rnn.seq_size do
			--Create new criterion for every timestep (CrossEntropy seems to be more robust than ClassNLL)
			local criterion=nn.CrossEntropyCriterion()--nn.ClassNLLCriterion()
			--First output is our prediction
			err[t]=criterion:forward(rnn[t].output[1],target[t])
			gradLoss[t]=criterion:backward(rnn[t].output[1],target[t])
		end
		--Initialize hidden gradient inputs for last timestep
		local hiddenGradient={}
		hiddenGradient[rnn.seq_size]={}
		for r=1,rnn.rnn_size do
			hiddenGradient[rnn.seq_size][r]=torch.zeros(rnn.hidden_size)
		end
		--Backward through time (in reverse to forward)
		for t=rnn.seq_size,1,-1 do
			--Pass input and gradient loss at timestep t
			rnn[t]:backward(input[t],{gradLoss[t],table.unpack(hiddenGradient[t])}) --TODO remove unpack call, it is terribly slow (>2 times)
			--We can get gradient inputs by calling rnn[t].gradInput
			--print(rnn[t].gradInput)
			--Initialize previous hidden gradient
			hiddenGradient[t-1]={}
			for r=1,rnn.rnn_size do
				hiddenGradient[t-1][r]=rnn[t].gradInput[r+1]:clone()
			end
		end
		--We update after batch of 40 iterations and after the last iteration in the current epoch
		if iteration%40==0 or iteration==#data-rnn.seq_size then
			rnn[1]:updateParameters(learning_rate)
			rnn[1]:zeroGradParameters()
			--Print error and sample generated text
			avg_err=0
			for every=1,#err do avg_err=avg_err+err[every] end
			avg_err=avg_err/#err
			--Print error after 1000 iteration
			if iteration%10000==0 then
				print("Error: ",avg_err)
				print(generateText(rnn,data.alphabet,50,"some seed"))
			end
		end
	end
	--Decrease learning rate after each epoch
	learning_rate=learning_rate*0.97
	print("Updating results.txt after "..counter..". epoch...")
	--Log match rate (measuring the quality of training)
	if counter%1==0 then
		local clock=os.clock()-time
		local rate=getMatchRate(rnn,data)
		local log=io.open("results.txt","a")
		log:write("-----------------------------------------------------------------\n")
		log:write("Average error after "..counter..". epoch = "..avg_err.."\n")
		log:write("Learning rate decayed to "..learning_rate.."\n")
		log:write("Epoch processed within "..clock.."s ("..(math.floor(clock/60)).."min "..(math.floor(clock%60)).."s)\n")
		log:write("Matching rate for training data = "..rate.." ("..(math.floor(rate*100)).."%)\n")
		log:close()
	end
	--make a snapshot (persist rnn)
	if counter%1==0 then
		local name="./snapshots/"..os.time().."ERROR"..avg_err..".rnn"
		torch.save(name,rnn)
		print("Snapshot of the RNN has been saved at "..name)
	end
end
end

function getMatchRate(rnn,data)
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
				hiddenState[t][r]=rnn[t].output[r+1]:clone()
			end
		end
		--Initialize target
		local target=data[iteration+rnn.seq_size]
		if TP.tensorToChar(target,data.alphabet)==TP.tensorToChar(rnn[rnn.seq_size].output[1],data.alphabet) then matched=matched+1 end
	end
	return matched/tests
end

local isize=#data.alphabet
local hsize=256
local rnn_size=3
local seq_size=20
local model=createModel(isize,hsize,rnn_size)
graph.dot(model.fg,"model","model")


local rnn=unfoldModel(model,seq_size)
trainUnfoldedModel(rnn,0.0002,data,1000)

--[[
rnn=torch.load("./snapshots/mynet.rnn")
print(generateText(rnn,data.alphabet,500,"seed"))
print(getMatchRate(rnn,data))
]]





--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--
--IT STAYS ONLY FOR REFERENCE
--NOT WORKING, FALLING BACK TO UNFOLDING NETWORK AND SHARING PARAMETERS
-- training data is an array of tensors received from TP method charToTensor which converts char into tensor based on the alphabet of the text
function trainModel(rnn,seq_size,data,learning_rate,epochs)--TODO batch training
	local err={}
	for counter=1,epochs do
		rnn:zeroGradParameters()--prevent stacking
		for i=1,#data-seq_size do
-------------------------------------FORWARDING------------------------------------------------
			local inputs={}
			local predictions={}
			--hidden states at zero timestep
			predictions[0]={}
			for r=1,rnn.rnn_size do
				predictions[0][r+1]=torch.zeros(rnn.hidden_size)
			end
			--do forward for every char in sequence and save predictions at each timestep
			for t=1,seq_size do
				--all inputs, except the first, are previous hidden states
				inputs[t]={}
				for r=1,rnn.rnn_size do
					inputs[t][r+1]=predictions[t-1][r+1]:clone()
				end
				inputs[t][1]=data[i+t-1]
				predictions[t]={}
				local outputs=rnn:forward(inputs[t])
				for r=1,rnn.rnn_size+1 do
					predictions[t][r]=outputs[r]:clone()
				end
			end
			inputs[seq_size+1]={}
			inputs[seq_size+1][1]=data[i+seq_size]--for target
--------------------------------------BACKWARDING------------------------------------------------
			local gradInputs={}
			--zero gradient inputs at maximum timestep
			gradInputs[seq_size]={}
			for r=1,rnn_size do
				gradInputs[seq_size][r+1]=torch.zeros(rnn.hidden_size)
			end
			--do backward for every char in sequence from the last one and save gradient input at each timestep
			for t=seq_size,1,-1 do
				local loss=nn.MSECriterion()
				err[t]=loss:forward(predictions[t][1],inputs[t+1][1])--returns error
				--print(err)
				gradInputs[t][1]=loss:backward(predictions[t][1],inputs[t+1][1])--returns gradient loss
				local gradients=rnn:backward(inputs[t],gradInputs[t])
				gradInputs[t-1]={}
				for r=1,rnn.rnn_size do
					gradInputs[t-1][r+1]=gradients[r+1]:clone()
				end
			end
			if i%40==0 or i==#data-seq_size then
				rnn:updateParameters(learning_rate)
				rnn:zeroGradParameters()
				local avg_err=0
				for every=1,#err do avg_err=avg_err+err[every] end
				print("Error: ",(avg_err/#err))
				print(generateText(rnn,rnn.hidden_size,rnn.rnn_size,seq_size,alphabet,50))
			end
		end
	end
	learning_rate=learning_rate*0.97
end

--trainModel(model,seq_size,data,0.0002,1000)
