
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.GloVeEmbedding'
require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-skip_unk',0,'whether to skip UNK tokens when sampling')
cmd:option('-input_loop',0,'whether to read new seed text from stdin after having finished sampling')
cmd:option('-word_level',0,'whether to operate on the word level, instead of character level (0: use chars, 1: use words)') --todo: set this in checkpoint
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
require 'util.SharedDropout'

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')


-- do a few seeded timesteps
local seed_text = opt.primetext

repeat

	local current_state
	current_state = {}
	for L = 1,checkpoint.opt.num_layers do
		-- c and h for all layers
		local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
		if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
		if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
		table.insert(current_state, h_init:clone())
		if checkpoint.opt.model == 'lstm' then
			table.insert(current_state, h_init:clone())
		end
	end
	state_size = #current_state
	if string.len(seed_text) > 0 then
		gprint('seeding with ' .. seed_text)
		gprint('--------------------------')
        local tokens = {}
        for c in CharSplitLMMinibatchLoader.tokens(seed_text, opt.word_level == 1) do
            if vocab[c] == nil then c = c:lower() end
            tokens[#tokens + 1] = c
        end
        --tokens[#tokens + 1] = '.'
        for _, c in ipairs(tokens) do --todo: word_level should be stored in checkpoint
            local idx = vocab[c]
            if idx ~= nil then
                prev_char = torch.Tensor{idx}
				if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
				if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
			local lst = protos.rnn:forward{prev_char, unpack(current_state)}
			-- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
			current_state = {}
			for i=1,state_size do table.insert(current_state, lst[i]) end
			prediction = lst[#lst] -- last element holds the log probabilities
		end
        end
	else
		-- fill with uniform probabilities over characters (? hmm)
		gprint('missing seed text, using uniform probability over first character')
		gprint('--------------------------')
		prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
		if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
		if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
	end

	-- start sampling/argmaxing
	for i=1, opt.length do

		-- log probabilities from the previous timestep
		if opt.sample == 0 then
			-- use argmax
            -- TODO: Skip UNK
			local _, prev_char_ = prediction:max(2)
			prev_char = prev_char_:resize(1)
		else
			-- use sampling
			prediction:div(opt.temperature) -- scale by temperature
			local probs = torch.exp(prediction):squeeze()
			probs:div(torch.sum(probs)) -- renormalize so probs sum to one

            if opt.skip_unk then
                prev_char = torch.multinomial(probs:float(), 2):float()
                prev_char = prev_char[1] == vocab["UNK"] and prev_char[{{2}}] or prev_char[{{1}}]
            else
				prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
			end
        end

		-- forward the rnn for next character
		local lst = protos.rnn:forward{prev_char, unpack(current_state)}
		current_state = {}
		for i=1,state_size do table.insert(current_state, lst[i]) end
		prediction = lst[#lst] -- last element holds the log probabilities

        word = ivocab[prev_char[1]]
        if opt.word_level and word == "RN" then 
            word = "\n"
        end
        io.write(word)
        if opt.word_level then
            io.write(" ")
        end
	end
	io.write('\n') io.flush()
    if opt.input_loop == 1 then
        seed_text = io.read()
    end
until opt.input_loop ~= 1

