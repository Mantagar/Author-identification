torch.setnumthreads(1) --after reinstalling torch script slowed 10 times, it brings back performance to about 2 times the time before upgrade
local TP=require 'data/text_processor.lua'
require 'nn'
require 'nngraph'
local RNN=require 'RNN'
local Colorizer=require 'Colorizer'
local Utils=require 'Utils'
--usage: th test.lua rnn_name path_to_answers
local rnn_name=arg[1]
local answers_path=arg[2]


--Load RNN
print(Colorizer.green("Loading network..."))
local rnn=RNN.loadSnapshot(rnn_name)
rnn.input_size=math.floor(rnn.input_size)
rnn.hidden_size=math.floor(rnn.hidden_size)
rnn.depth=math.floor(rnn.depth)
rnn.heads=math.floor(rnn.heads)

--Display characteristics
print(Colorizer.yellow("RNN:\t\t"),rnn_name)
print(Colorizer.yellow("INPUT/OUTPUT SIZE:"),rnn.input_size)
print(Colorizer.yellow("HIDDEN SIZE:\t"),rnn.hidden_size)
print(Colorizer.yellow("RNN SIZE (STACKS):"),rnn.depth)
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
local avg_matches=torch.zeros(rnn[1].heads)
local correct=0
local matches={}
print(Colorizer.green("Calculating average performance..."))
for i=1,#data do
		print("  Calculating scores for the unknown text: ",i)
		matches[i]=Utils.getMatchRate(rnn,data[i][1],alphabet)
		avg_matches=avg_matches+matches[i]
end
avg_matches=avg_matches/rnn[1].heads
--print results for every unknown text
for i=1,#data do
		print(Colorizer.yellow("\nUnknown for author:"),i)
		print(Colorizer.yellow("Correct answer:\t"),answers[i])
		local size=#data

    local prob=matches[i]-avg_matches
		local min,best=prob:min(1)
		prob=prob-min[1]
		local max,_=prob:max(1)
		prob=1-prob/max[1]

    --print scores for all heads display
		for k=1,size do
			if k==i then
				print(Colorizer.white(" "..prob[k]))
			else
				print(" "..prob[k])
			end
		end

    --print score for only the questioned head
    --print(Colorizer.white(" "..prob[i]))
		if (i==best[1] and answers[i]==true) or (i~=best[1] and answers[i]==false) then
			print(Colorizer.green("Verification successful"))
			correct=correct+1
		else
			print(Colorizer.red("Verification failed"))
		end
end
--Print final outcome
print(Colorizer.green("\nPreparing final outcome...")) 
print(Colorizer.yellow("Overall verification score:"),correct/rnn.heads)
