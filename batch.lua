--Stochastic Gradient Descent
for i=1,m do --for every sample update and zero
	local pred = net:forward(x[i]) --pass input to the network
	local err = loss:forward(pred,y[i]) --pass prediction and output to the criterion
	local gradLoss = loss:backward(pred,y[i]) --backward pass
	net:zeroGradParameters() --zero net gradients every time
	net:backward(x[i],gradLoss) --backward pass through network
	net:updateParameters(learning_rate) --update weights with learning rate
end

-- Mini-Batch Gradient Descent

for i =1,m,batchSize do
	net:zeroGradParameters() --zero only at the begining
	for j=0,batchSize-1 do --for every sample in batch
		if i+j>m then break end
		local pred = net:forward(x[i+j]) --pass input to the network
		local err = loss:forward(pred,y[i+j]) --pass prediction and output to the criterion
		local gradLoss = loss:backward(pred,y[i+j]) --backward pass
		net:backward(x[i+j],gradLoss) --backward pass through network
	end
	net:updateParameters(learning_rate) --update after processing a whole batch
end
