local BatchManager={}

function BatchManager.createBatcher(data,batch_size,input_size)
	local batcher={
		rows=batch_size,--minibatch size
		batch_data=data,
		cols=input_size,--normal input size
		pos=1--current position in batch
	}
	return batcher
end

function BatchManager.isNextBatch(batcher)
 return batcher.pos<=#batcher.batch_data
end

--[[
batcher - table created with createBatcher function
seq_size - how many times in time rnn was unfolded
]]
function BatchManager.nextBatch(batcher,seq_size)
	local inputs={}
	local labels={}
	local starting_pos=batcher.pos
	for i=1,seq_size do
		inputs[i]=torch.DoubleTensor(batcher.rows,batcher.cols)
		labels[i]=torch.DoubleTensor(batcher.rows,batcher.cols)
    for r=1,batcher.rows do
			if batcher.pos+i+r-2<#batcher.batch_data then
				inputs[i][r]:copy(batcher.batch_data[batcher.pos+i+r-2])
				labels[i][r]:copy(batcher.batch_data[batcher.pos+i+r-1])
			else
				inputs[i][r]=torch.zeros(batcher.cols)
				labels[i][r]=torch.zeros(batcher.cols)
			end
		end
	end
	batcher.pos=batcher.pos+batcher.rows
	return inputs,labels
end

--here we can test that it actually works
function BatchManager.test()
local seq_size=5
	local data={}
	for i=1,10 do
		data[i]=torch.Tensor{i,i,i,i}
	end
	local batcher=BatchManager.createBatcher(data,5,4)
	while BatchManager.isNextBatch(batcher) do
		local in1,out1=BatchManager.nextBatch(batcher,seq_size)
		for i=1,seq_size do
			print(in1[i])
		end
		for i=1,seq_size do
			print(out1[i])
		end
	end
end

return BatchManager
