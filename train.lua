torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'data/text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'
local Colorizer=require 'Colorizer'
local Utils=require 'Utils'

--parse command line
cmd = torch.CmdLine()
cmd:text()
cmd:text(Colorizer.yellow('Run to train multi-headed, character-level, deep RNN'))
cmd:text()
cmd:text(Colorizer.yellow('Configuration:'))
cmd:option('-max_authors',10,'maximum number of authors')
cmd:option('-hidden_size',199,'neurons in the hidden layers')
cmd:option('-depth',3,'stacks in deep RNN')
cmd:option('-unroll_times',20,'length of character sequences')
cmd:option('-max_epochs',10,'maximum number of cycles over the data')
cmd:option('-learning_rate',0.02,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay after each epoch')
cmd:option('-batch_size',40,'size of the input batch')
cmd:text()
config = cmd:parse(arg)

--load data
print(Colorizer.green("Loading data..."))
local authors=Utils.getKnownData(config.max_authors)
if #authors==0 then
	print(Colorizer.red("No data availible!"))
	return
end
print(Colorizer.green("Begining the training..."))

--create model
local input_size=#authors[1][1].alphabet--get size of the first document's alphabet 
local model=RNN.createModel(input_size,config.hidden_size,config.depth,#authors)

--generate graf representation of model
graph.dot(model.fg,"multihead-model","mulithead-model")

--unroll in time
local rnn=RNN.unrollModel(model,config.unroll_times)

--train RNN
for i=1,config.max_epochs do
	for author_no=1,#authors do
		for doc_no=1,#authors[author_no] do
			print(Colorizer.yellow("Epoch "..i.."/"..config.max_epochs.."\t".."Author "..author_no.."/"..#authors.."\t".."Document "..doc_no.."/"..#authors[author_no]))
			RNN.trainUnrolledModel(rnn,config.learning_rate,authors[author_no][doc_no],author_no,config.batch_size)
		end
	end
	if i%5==0 then
		local mem=collectgarbage("count")
		collectgarbage()
		print(Colorizer.green("Garbage collector run."))
		print(Colorizer.yellow("Memory before:"),mem)
		print(Colorizer.yellow("Memory after:"),collectgarbage("count"))
	end
	config.learning_rate=config.learning_rate*config.learning_rate_decay
end

--save RNN
print(Colorizer.green("Learning finished"))
print(Colorizer.green("Saving snapshot..."))
RNN.makeSnapshot(rnn)

