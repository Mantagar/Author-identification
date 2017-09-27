local RNN={}

local BatchManager=require 'BatchManager'

--Creates model of multiheaded reccurent neural network
function RNN.createModel(input_size,hidden_size,depth,heads)
	local input={}
	local hidden={}
	local output={}
	input[1]=nn.Identity()()
	for i=2,depth+1 do
		input[i]=nn.Identity()()
		local i2h = input[i] - nn.Linear(hidden_size,hidden_size)
		local h2h = i==2 and input[1] - nn.Linear(input_size,hidden_size) or hidden[i-2] - nn.Linear(hidden_size,hidden_size)
		hidden[i-1] = {h2h,i2h} - nn.CAddTable() - nn.Tanh()
		output[heads+i-1] = hidden[i-1]
	end
	for i=1,heads do
		output[i]=hidden[depth] - nn.Linear(hidden_size,input_size) --outputs
	end
	--ANNOTATIONS
	for i,each in ipairs(input) do
			each:annotate{name='HIDDEN STATE[t-1]['..(i-1)..']\n', graphAttributes = {style="filled", fillcolor = '#aaaaff'}}
	end
	input[1]:annotate{name='INPUT\n', graphAttributes = {style="filled", fillcolor = '#ffffaa'}}
	for i,each in ipairs(hidden) do
			each:annotate{name='HIDDEN STATE[t]['..i..']\n', graphAttributes = {style="filled", fillcolor = '#ffaaaa'}}
	end
	for i=1,heads do
		output[i]:annotate{name='OUTPUT\nHEAD['..i..']\n', graphAttributes = {style="filled", fillcolor = '#aaffaa'}}
	end

	model=nn.gModule(input,output)
	model.input_size=input_size
	model.hidden_size=hidden_size
	model.depth=depth
	model.heads=heads
	return model
end

--Unrolls model in time for sequences of length unroll_times and returns appropriate RNN
function RNN.unrollModel(model,unroll_times)
	--Clone models in time
	rnn={}
	for i=1,unroll_times do
		rnn[i]=model:clone()
	end
	--Share weight, bias, gradWeight and gradBias
	local sharedParams,sharedGradParams=rnn[1]:parameters()
	for t=2,unroll_times do
		local params,gradParams=rnn[t]:parameters()
		for i=1,#params do
			params[i]:set(sharedParams[i])
			gradParams[i]:set(sharedGradParams[i])
		end
	end
	rnn.unroll_times=unroll_times
	rnn.hidden_size=model.hidden_size
	rnn.input_size=model.input_size
	rnn.depth=model.depth
	rnn.heads=model.heads

	return rnn
end

--TRAIN UNROLLED MODEL (one epoch for the chosen author)
function RNN.trainUnrolledModel(rnn,learning_rate,data,head_no,batch_size)
	--Optimization constants
	local ZERO_HEAD_VECTOR=torch.zeros(batch_size,rnn.input_size) --used for gradient inputs of other heads
	local ZERO_HIDDEN_VECTOR=torch.zeros(batch_size,rnn.hidden_size) --used for first hidden state input and last hidden state gradient input
	--Batcher preparation
	local batcher=BatchManager.createBatcher(data,batch_size,rnn.input_size)
	--Zero initial gradient
	rnn[1]:zeroGradParameters()
	--If data availible then feed it to the network
	while BatchManager.isNextBatch(batcher) do
		--Initialize the initial hidden state with zero matrices
		local hiddenState={}
		hiddenState[0]={}
		for r=1,rnn.depth do
			hiddenState[0][r]=ZERO_HIDDEN_VECTOR --passing only reference
		end
		--Initialize minibatches at consecutive timesteps
		local input,label=BatchManager.nextBatch(batcher,rnn.unroll_times)
		--Forward through time
		for t=1,rnn.unroll_times do
			local inputAndHidden={}
			inputAndHidden[1]=input[t]
			for r=1,rnn.depth do
				inputAndHidden[r+1]=hiddenState[t-1][r]
			end
			--Pass input and previous hidden state at timestep t
			rnn[t]:forward(inputAndHidden)
			--We can get the outputs by calling rnn[t].output
			--Initialize next hidden state
			hiddenState[t]={}
			for r=1,rnn.depth do
				hiddenState[t][r]=rnn[t].output[rnn.heads+r]:clone()
			end
		end
		--Initialize targets (for CrossEntropyCriterion instead of providing array we provide the index of one_hot's 1)
		local target={}
		for t=1,rnn.unroll_times do
			target[t]=torch.Tensor(batch_size)
			for b=1,batch_size do
				_,index=label[t][b]:max(1)
				target[t][b]=index
			end
		end
		--Calculate error and gragient loss at every timestep
		local err={}
		local gradLoss={}
		for t=1,rnn.unroll_times do
			--Create new criterion for every timestep
			local criterion=nn.CrossEntropyCriterion()
			--First output is our prediction
			err[t]=criterion:forward(rnn[t].output[head_no],target[t])
			gradLoss[t]=criterion:backward(rnn[t].output[head_no],target[t])
		end
		--Initialize hidden gradient inputs for last timestep
		local hiddenGradient={}
		hiddenGradient[rnn.unroll_times]={}
		for r=1,rnn.depth do
			hiddenGradient[rnn.unroll_times][r]=ZERO_HIDDEN_VECTOR --passing only reference
		end
		--Initialize gradient inputs for heads (all timesteps)
		local headsGradient={}
		for t=1,rnn.unroll_times do
			headsGradient[t]={}
			for h=1,rnn.heads do
				headsGradient[t][h]=ZERO_HEAD_VECTOR --passing only reference
			end
			headsGradient[t][head_no]=gradLoss[t]
		end
		--Backward through time (in reverse to forward)
		for t=rnn.unroll_times,1,-1 do
			--Pass input and gradient loss at timestep t
			local gradient={}
			for h=1,rnn.heads do
				gradient[h]=headsGradient[t][h]
			end
			for r=1,rnn.depth do
				gradient[#gradient+1]=hiddenGradient[t][r]
			end
			rnn[t]:backward(input[t],gradient)
			--We can get gradient inputs by calling rnn[t].gradInput
			--Initialize previous hidden gradient
			hiddenGradient[t-1]={}
			for r=1,rnn.depth do
				hiddenGradient[t-1][r]=rnn[t].gradInput[r+1]:clone()
			end
		end
		--We update after every minibatch
		rnn[1]:updateParameters(learning_rate)
		rnn[1]:zeroGradParameters()
		local avg_err=0
		for every=1,#err do avg_err=avg_err+err[every] end
		avg_err=avg_err/#err
		print("Error for author:",head_no,avg_err)
	end
end

function RNN.makeSnapshot(rnn)
	local path="./snapshots/"..os.time()..".rnn"
	torch.save(path,rnn)
	print("Snapshot saved at "..path)
end

function RNN.loadSnapshot(name)
	local path="./snapshots/"..name..".rnn"
	local rnn=torch.load(path)
	return rnn
end


return RNN
