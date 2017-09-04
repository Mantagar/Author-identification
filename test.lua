torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'data/text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'
local Colorizer=require 'Colorizer'
local Utils=require 'Utils'
--usage: th test.lua rnn_name.rnn path_to_answers
local rnn_name=arg[1]
local answers_path=arg[2]


--Load RNN
print(Colorizer.green("Loading network..."))
local rnn=RNN.loadSnapshot(rnn_name)
rnn.input_size=math.floor(rnn.input_size)
rnn.hidden_size=math.floor(rnn.hidden_size)
rnn.rnn_size=math.floor(rnn.rnn_size)
rnn.heads=math.floor(rnn.heads)

--Display characteristics
print(Colorizer.yellow("RNN:\t\t"),rnn_name)
print(Colorizer.yellow("INPUT/OUTPUT SIZE:"),rnn.input_size)
print(Colorizer.yellow("HIDDEN SIZE:\t"),rnn.hidden_size)
print(Colorizer.yellow("RNN SIZE (STACKS):"),rnn.rnn_size)
print(Colorizer.yellow("HEADS:\t\t"),rnn.heads)

--Load answers
print(Colorizer.green("Loading answers..."))
local answers=Utils.loadAnswers(answers_path,rnn.heads)
print(answers)


--Load unknown data
print(Colorizer.green("Loading data..."))
local data=Utils.getUnknownData(rnn.heads)
if #data==0 then
	print(Colorizer.red("No data availible!"))
	return
end
local alphabet=data[1][1].alphabet

--Verify texts
local correct=0
for i=1,#data do
		print(Colorizer.green("\nCalculating match rates..."))
		local matches=Utils.getMatchRate(rnn,data[i][1],alphabet)
		print(Colorizer.yellow("Unknown for author:"),i)
		print(Colorizer.yellow("Correct answer:\t"),answers[i])
		local _,max_index=matches:max(1)
		max_index=max_index[1]
		print(Colorizer.yellow("Best fit for author:"),max_index)
		local size=(#matches)[1]
		local average=matches:sum()/size
		print(Colorizer.yellow("Average score:"),average)
		for k=1,size do
			if k==i then
				print(Colorizer.red(" "..matches[k]))
			else
				print(" "..matches[k])
			end
		end
		if (average<matches[i] and answers[i]==true) or (average>matches[i] and answers[i]==false) then
			print(Colorizer.yellow("Verification successful"))
			correct=correct+1
		else
			print(Colorizer.yellow("Verification failed"))
		end
end
--Print final outcome
print(Colorizer.green("Preparing final outcome...")) 
print(Colorizer.yellow("Overall verification score:"),correct/rnn.heads)
