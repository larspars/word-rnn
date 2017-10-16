
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'pprint'

require 'util.OneHot'
require 'util.GloVeEmbedding'
require 'util.misc'
require 'util.SharedDropout'
require 'util.Zoneout'
require 'util.LayerNormalization'
require 'util.LookupTableFixed'

local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'
local IRNN = require 'model.IRNN'
local SDRNN = require 'model.SDRNN'
local lsuvInit = require 'util.LsuvInit'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-embedding_size', 200, 'size of word embeddings')
cmd:option('-learn_embeddings', 1, '1=learn embeddings, 0=keep embeddings fixed')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-num_fixed', 0 ,'number of recurrent layers to remain fixed (untrained), pretrained (LSTM only)')
cmd:option('-model', 'lstm', 'lstm, gru, rnn or irnn')
cmd:option('-lsuv_init', 0, 'use layer-sequential unit-variance (LSUV) initialization ')
cmd:option('-multiplicative_integration', 0, 'turns on multiplicative integration (as opposed to simply summing states)')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',1,'learning rate decay (rmsprop only)')
cmd:option('-learning_rate_decay_after',0,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_by_val_loss',0,'if 1, learning rate is decayed when a validation loss is not smaller than the previous')
cmd:option('-learning_rate_decay_wait',0,'the minimum number of epochs the learning rate is kept after decaying it because of validation loss')
cmd:option('-decay_rate',0.5,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-recurrent_dropout',0,'dropout for regularization, used on recurrent connections. 0 = no dropout')
cmd:option('-zoneout',0,'zoneout for regularization, used on recurrent connections. 0 = no zoneout')
cmd:option('-zoneout_c',0,'zoneout on the lstm cell. 0 = no zoneout')
cmd:option('-recurrent_depth', 0, 'the number of additional h2h matrices, when the model is an SDRNN')
cmd:option('-gradient_noise',0,'amount of gradient noise for regularization (will be decayed over time t, as b/t^0.55 )')
cmd:option('-activation_clamp',0,'clamp activations at this value (sdrnn only)')
cmd:option('-activation_l2',0,'amount of l2 penalization to apply to the activations (sdrnn only)')
cmd:option('-l2',0,'amount of l2 weight decay to regularize the model with')
cmd:option('-activation_l1',0,'amount of l1 weight decay to regularize the model with (rnn & dfarnn only)')
cmd:option('-batch_normalization',0,'whether to apply batch normalization (0=no BN, 1=vertical BN, 2=vertical and horizontal BN)')
cmd:option('-layer_normalization',0,'whether to apply layer normalization')

cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-max_norm',0,'make sure gradient norm does not exceed this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-random_crops', 1, 'use a random crop of the training data per epoch when it does not evenly divide into the number of batches')

-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-word_level',0,'whether to operate on the word level, instead of character level (0: use chars, 1: use words)')
cmd:option('-threshold',0,'minimum number of occurences a token must have to be included (ignored if -word_level is 0)')
cmd:option('-glove',0,'whether or not to use GloVe embeddings')
cmd:option('-non_glove_embedding',0,'use embeddings, with random intialization')
cmd:option('-optimizer','rmsprop','which optimizer to use: adam or rmsprop')


cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.word_level == 1, opt.threshold)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate }
local skip_iterations = 0
local train_losses = {}
local latest_train_losses = {}

if opt.eval_val_every == -1 then
    opt.eval_val_every = loader.ntrain
end

val_losses = {}
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end


function gpu_prepare(tensor)
    if opt.gpuid >=0 and opt.opencl == 0 then tensor = tensor:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then tensor = tensor:cl() end
    return tensor
end

function extractLayers(modules)
   local layers = {}
   for k,v in pairs(modules) do
      if v.modules then
         for k2,v2 in pairs(extractLayers(v.modules)) do
            table.insert(layers, v2)
         end
      else
         table.insert(layers, v)
      end
   end
   return layers
end

-- define the model: prototypes for one timestep, then clone them in time
local h2hs = nil
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    optim_state = checkpoint.optim_state
    init_state_global = checkpoint.init_state_global
    if opt.overwrite_learning_rate == 1 then
        optim_state.learningRate = opt.learning_rate
        print("learning rate set to ", optim_state.learningRate)
    end
    print(optim_state)
    -- make sure the vocabs are the same
    local vocab_compatible = true
    local checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i) then
            vocab_compatible = false
        end
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    end
    for k,v in pairs(extractLayers(checkpoint.protos.rnn.modules)) do
        if torch.typename(v) == 'nn.SharedDropout' then
            nn.SharedDropout.init(v.id, v.p)
        end
    end
    if not (checkpoint_vocab_size == vocab_size) then
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    print(checkpoint.opt)
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.optimizer = checkpoint.optimizer
    opt.model = checkpoint.opt.model
    skip_iterations = checkpoint.i
    local init_dir = string.sub(opt.init_from, 1, string.find(opt.init_from, "/[^/]*$"))
    local lossfile = init_dir .. "trainlosses.t7"
    if path.exists(lossfile) then
        train_losses = torch.load(lossfile)
    end
    val_losses = checkpoint.val_losses
else
    print(opt.checkpoint_dir)
    local file = io.open(string.format('%s/options.txt', opt.checkpoint_dir), "w")
    file:write(pprint.pretty_string(opt))
    file:close()

     print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    local embedding = nil
    if opt.glove == 1 then
        if opt.learn_embeddings == 0 then
            embedding = GloVeEmbeddingFixed(vocab, opt.embedding_size, opt.data_dir, "", true)
        else
            embedding = GloVeEmbedding(vocab, opt.embedding_size, opt.data_dir, "", true)
        end
    elseif opt.non_glove_embedding ~= 0 then
        if opt.learn_embeddings == 0 then
            print("using fixed embeddings")
            embedding = LookupTableFixed(vocab_size, opt.embedding_size)
        else
            embedding = nn.LookupTable(vocab_size, opt.embedding_size)
        end
    end
    input_size = (opt.glove == 1 or opt.non_glove_embedding ~= 0) and opt.embedding_size or vocab_size
    if opt.model == 'lstm' then
	protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.recurrent_dropout, embedding, opt.num_fixed)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, embedding)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, embedding)
    elseif opt.model == 'irnn' then
        protos.rnn, h2hs = IRNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, embedding)
    elseif opt.model == 'sdrnn' then
        protos.rnn = SDRNN.sdrnn(input_size, vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, embedding)
    end
    protos.criterion = nn.ClassNLLCriterion()

    --local clusters = {}
    --for w,i in pairs(vocab) do 
    --    clusters[#clusters+1] = {1, i}
    --end
    --protos.criterion = nn.HSM(torch.Tensor(clusters), opt.rnn_size, 0) --vocab['UNK'])
end

print('using optimizer ' .. opt.optimizer)
-- the initial state of the cell/hidden states
init_state = {}
for L=1, #protos.rnn.outnode.data.mapindex-1 do
    local size = opt.rnn_size
    table.insert(init_state, gpu_prepare(torch.zeros(opt.batch_size, size)))

end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

if opt.lsuv_init == 1 and string.len(opt.init_from) == 0 then
    nn.SharedDropout.reset()
    lsuvInit(protos.rnn, vocab_size)
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' and string.len(opt.init_from) == 0 then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx and layer_idx > opt.num_fixed then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        if i % opt.print_every == 0 then
            print(i .. '/' .. n .. '... (' ..  (loss / opt.seq_length / i) .. ')')
        end
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

function ifelse(condition, a, b)
    if condition then return a else return b end
end

function table.slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end

function saveJson(filename, obj)
    local file = io.open(filename, "w")
    file:write(pprint.pretty_string(obj))
    file:close()
end

local shrank_norm = false
-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    if opt.optimizer == "sgd" or opt.optimizer == "adam" then
        grad_params:div(opt.seq_length)
    end
    if opt.gradient_noise ~= 0 then
        noise = noise or torch.Tensor():typeAs(grad_params):resizeAs(grad_params)
        local stddev = math.sqrt(opt.gradient_noise/(b^0.55))
        noise:normal(0, stddev)
        grad_params:add(noise)
    end
    -- clip gradient element-wise
    if opt.grad_clip ~= 0 then
        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    end
    if opt.max_norm ~= 0 then
        local norm = grad_params:norm()
        if norm > opt.max_norm  then
            shrank_norm = true
            grad_params:mul(opt.max_norm / norm)
        end
    end
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

local optimizer = nil

if opt.optimizer == 'adam' then
    optimizer = optim.adam
elseif opt.optimizer == 'sgd' then
    optimizer = optim.sgd
    optim_state.learningRateDecay = opt.decay_rate
    optim_state.momentum = 0.99
    optim_state.nesterov = true
    optim_state.dampening = 0
else
    optimizer = optim.rmsprop
end


local trainerr_interval = math.floor(2000 / opt.seq_length) --print avg loss this often
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()

    local _, loss = optimizer(feval, params, optim_state)

    if opt.eval_test == 1 then
        nn.SharedDropout.reset()
        local test_loss = eval_split(3, math.huge)
        print("Test loss:", test_loss, "BPC:", test_loss/math.log(2))
        return
    end

    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    table.insert(latest_train_losses, train_loss)


    -- exponential learning rate decay for rmsprop
    if opt.optimizer == 'rmsprop' and i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after and opt.learning_rate_decay_after ~= 0 then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end
    if i % trainerr_interval == 0 then
        local sum = 0
        for m=1,#latest_train_losses do
            sum = sum + latest_train_losses[m]
        end
        print("Avg train error", sum/#latest_train_losses)
        table.insert(train_losses, sum/#latest_train_losses)
        latest_train_losses = {}
    end

    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data

        local max_batches = 50000000 --math.floor(400 * 256 / opt.batch_size)
        local val_loss = eval_split(2, max_batches) -- 2 = validation
        val_losses[i] = val_loss

        local prev_val_loss = val_losses[i-opt.eval_val_every] or 0
        if i >= opt.eval_val_every*4 and prev_val_loss <= val_loss
            and opt.learning_rate_decay_by_val_loss == 1 and epoch >= lastDecayEpoch + opt.learning_rate_decay_wait then
            optim_state.learningRate = optim_state.learningRate * opt.decay_rate
            lastDecayEpoch = math.floor(epoch + 0.5)
            print("Decayed learning rate to ", optim_state.learningRate)
        end

        local sum = 0
        local trainlosses_to_avg = 3
        for m=#train_losses, #train_losses-trainlosses_to_avg+1, -1 do
            sum = sum + (train_losses[m] or 0)
        end
        local iters = math.min(opt.eval_val_every, i)
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f__t%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss, sum/trainlosses_to_avg)
        local lossfile = string.format('%s/trainlosses.t7', opt.checkpoint_dir)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        checkpoint.optim_state = optim_state
        checkpoint.optimizer = opt.optimizer
        checkpoint.init_state_global = init_state_global
        checkpoint.horizontal_bn = {}
        for t=1, opt.seq_length do
            --todo: make use of these when initing
            checkpoint.horizontal_bn[t] = {}
            for _,node in ipairs(clones.rnn[t].forwardnodes) do
                if torch.typename(node.data.module) == 'nn.BatchNormalization' then
                    table.insert(checkpoint.horizontal_bn[t], node.data.module)
                end
            end
        end
        collectgarbage()
        torch.save(savefile, checkpoint)
        torch.save(lossfile, train_losses)
    end

    if i % opt.print_every == 0 then
        --print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        
        printStr = string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/1000 = %.4fs", i, iterations, epoch, train_loss, time)
        if opt.max_norm ~= 0 and shrank_norm then
            printStr = printStr .. ", norm=" .. opt.max_norm
        end
        if i % (opt.print_every*20) == 0 then
            printStr = printStr .. string.format(", gn/bn = %6.8f", grad_params:norm() / params:norm())--this is a little expensive, so do it more rarely
            --printStr = printStr .. string.format(", gn = %6.8f", grad_params:norm())
        end
        print(printStr)
    end
      
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if i % loader.ntrain == 0 then
        print("zeroing random state")
        for k,v in pairs(init_state_global) do 
            v:zero()
        end 
        --[[
        print("setting random state with unit variance")
        for k,v in pairs(init_state_global) do 
            v:normal(0, 1)
            v:div(v:std()) --random noise with unit variance
        end 
        ]]--
    end
    if loss[1] > loss0 * 100 then
        print(string.format("loss is exploding, aborting. (%6.2f vs %6.2f)", loss0, loss[1]))
        break -- halt
    end
end


