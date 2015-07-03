
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
cmd:option('-primetext'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-skip_unk',0,'whether to skip UNK tokens when sampling')
cmd:option('-word_level',0,'whether to operate on the word level, instead of character level (0: use chars, 1: use words)') --todo: set this in checkpoint
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)


local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

protos = checkpoint.protos
local rnn_idx = #protos.softmax.modules - 1
opt.rnn_size = protos.softmax.modules[rnn_idx].weight:size(2)

-- initialize the rnn state
local current_state, state_predict_index
local model = checkpoint.opt.model

print('creating an LSTM...')
local num_layers = checkpoint.opt.num_layers or 1 -- or 1 is for backward compatibility
current_state = {}
for L=1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
state_predict_index = #current_state -- last one is the top h
local seed_text = opt.primetext
local prev_char

protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- do a few seeded timesteps
print('seeding with ' .. seed_text)
for c in CharSplitLMMinibatchLoader.tokens(seed_text, opt.word_level == 1) do --todo: word_level should be stored in checkpoint
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end
end

-- start sampling/argmaxing
for i=1, opt.length do

    -- softmax from previous timestep
    local next_h = current_state[state_predict_index]
    next_h = next_h / opt.temperature
    local log_probs = protos.softmax:forward(next_h)

    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = log_probs:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        local probs = torch.exp(log_probs):squeeze()
        if opt.skip_unk then
            prev_char = torch.multinomial(probs:float(), 2):float()
            prev_char = prev_char[1] == vocab["UNK"] and prev_char[{{2}}] or prev_char[{{1}}]
        else
            prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
        end
    end

    -- forward the rnn for next character
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end

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

