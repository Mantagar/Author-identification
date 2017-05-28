function createModel(input_size,hidden_size,rnn_size,output_amount)
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
			each:annotate{name='HIDDEN STATE[t-1]['..(i-1)..']\n', graphAttributes = {style="filled", fillcolor = 'lightblue'}}
	end
	input[1]:annotate{name='INPUT[t]\n', graphAttributes = {style="filled", fillcolor = 'yellow'}}
	for i,each in ipairs(hidden) do
			each:annotate{name='HIDDEN STATE[t]['..i..']\n', graphAttributes = {style="filled", fillcolor = 'red'}}
	end
	for i=1,output_amount do
		output[i]:annotate{name='OUTPUT[t]\nAUTHOR['..i..']\n', graphAttributes = {style="filled", fillcolor = 'yellow'}}
	end
	print("INPUT: "..#input,"HIDDEN: "..#hidden,"OUTPUT: "..#output)

	model=nn.gModule(input,output)
	model.input_size=input_size
	model.hidden_size=hidden_size
	model.rnn_size=rnn_size
	model.output_amount=output_amount
	return model
end


require 'nn'
require 'nngraph'
local isize=50
local hsize=256
local rnn_size=2
local seq_size=20
local output_amount=3
local model=createModel(isize,hsize,rnn_size,output_amount)
graph.dot(model.fg,"multihead-model","mulithead-model")

