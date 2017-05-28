local RNN={}
--Creates model of multiheaded reccurent neural network
function RNN.createModel(input_size,hidden_size,rnn_size,output_amount)
	local input={}
	local hidden={}
	local output={}
	input[1]=nn.Identity()()
	for i=2,rnn_size+1 do
		input[i]=nn.Identity()()
		local i2h = input[i] - nn.Linear(hidden_size,hidden_size)
		local h2h = i==2 and input[1] - nn.Linear(input_size,hidden_size) or hidden[i-2] - nn.Linear(hidden_size,hidden_size)
		hidden[i-1] = {h2h,i2h} - nn.CAddTable() - nn.Tanh()
		output[output_amount+i-1] = hidden[i-1]
	end
	for i=1,output_amount do
		output[i]=hidden[rnn_size] - nn.Linear(hidden_size,input_size) - nn.LogSoftMax()
	end
	--ANNOTATIONS
	--TODO remove all ipairs calls, they are terriblly slow
	for i,each in ipairs(input) do
			each:annotate{name='HIDDEN STATE[t-1]['..(i-1)..']\n', graphAttributes = {style="filled", fillcolor = '#aaaaff'}}
	end
	input[1]:annotate{name='INPUT[t]\n', graphAttributes = {style="filled", fillcolor = '#ffffaa'}}
	for i,each in ipairs(hidden) do
			each:annotate{name='HIDDEN STATE[t]['..i..']\n', graphAttributes = {style="filled", fillcolor = '#ffaaaa'}}
	end
	for i=1,output_amount do
		output[i]:annotate{name='OUTPUT[t]\nAUTHOR['..i..']\n', graphAttributes = {style="filled", fillcolor = '#aaffaa'}}
	end
	print("INPUT: "..#input,"HIDDEN: "..#hidden,"OUTPUT: "..#output)

	model=nn.gModule(input,output)
	model.input_size=input_size
	model.hidden_size=hidden_size
	model.rnn_size=rnn_size
	model.output_amount=output_amount
	return model
end

--Unfolds model in time for sequences of length seq_size and returns appropriate RNN
function RNN.unfoldModel(model,seq_size)
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
	rnn.output_amount=model.output_amount
	return rnn
end

--TODO data needs proper format [author->documents] zero gradients should be fed to remaining authors during backpropagation
--TRAIN UNFOLDED MODEL
function RNN.trainUnfoldedModel(rnn,learning_rate,data,epochs)
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

require 'nn'
require 'nngraph'
local isize=50
local hsize=256
local rnn_size=2
local seq_size=20
local output_amount=3
local model=RNN.createModel(isize,hsize,rnn_size,output_amount)
graph.dot(model.fg,"multihead-model","mulithead-model")


return RNN
