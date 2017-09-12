
require 'util.StochasticSkip'
require 'util.L2Penalty'
require 'util.BELU'
require 'util.BReLU'


local nninit = require 'util.nninit_monkeypatch'

local SDRNN = {}


function SDRNN.sdrnn(input_size, vocab_size, rnn_size, total_num_layers, dropout, embedding)
  --nngraph.setDebug(true)
  local function summation(size, L, y1, y2)
    return nn.CAddTable(true)({y1, y2})
  end

  local function summation_w_bias(size, L, y1, y2)
    return nn.Add(size):init('bias', nninit.constant, 0)(nn.CAddTable(true)({y1, y2}))
  end

  local function multiplicative_integration(size, L, y1, y2)
    --a*y1*y2 + b1*y2 + b2*y1 + b3
    return nn.Add(size):init('bias', nninit.constant, 0)(  --b3
                nn.CAddTable()({
                  nn.CMulTable()({  --(a*y1)*y2
                        nn.CMul(size):init('weight', nninit.constant, 1)(y1), --a*y1
                        y2
                  }):annotate{name='ay1y2_'..L},  --*y2
                  nn.CMul(size):init('weight', nninit.constant, 0.5)(y2):annotate{name='b1y2_'..L}, -- b1*y2
                  nn.CMul(size):init('weight', nninit.constant, 0.5)(y1):annotate{name='b2y1_'..L}, -- b2*y1
                })
           ):annotate{name='b3_'..L}
  end

  local function simple_multiplicative_integration(size, L, y1, y2)
    --y1*y2 + b
    return nn.CMulTable()({ y1, y2 })
  end

  local function multiplicative_integration_clamped(size, L, y1, y2)
    --tanh(a*y1*y2) + b1*y2 + b2*y1 + b3
    return nn.Add(size):init('bias', nninit.constant, 0)(  --b3
                nn.CAddTable(true)({
                  nn.Clamp(-2, 2)(
                    nn.CMulTable()({  --(a*y1)*y2
                        nn.CMul(size):init('weight', nninit.constant, 1)(y1), --a*y1
                        y2
                    }):annotate{name='ay1y2_'..L}
                  ),  --*y2
                  nn.CMul(size):init('weight', nninit.constant, 1)(y2):annotate{name='b1y2_'..L}, -- b1*y2
                  nn.CMul(size):init('weight', nninit.constant, 1)(y1):annotate{name='b2y1_'..L}, -- b2*y1
                })
           ):annotate{name='b3_'..L}
  end

  local function summation_w_gain(size, L, y1, y2)
    return nn.Add(size):init('bias', nninit.constant, 0)(
            nn.CMul(size):init('weight', nninit.constant, 1)(nn.CAddTable(true)({y1, y2}))
           )
  end

  local combiner = summation
  if opt.multiplicative_integration == 1 then
    combiner = multiplicative_integration
  elseif opt.multiplicative_integration == 2 then
    combiner = simple_multiplicative_integration
  elseif opt.multiplicative_integration == 3 then
    combiner = multiplicative_integration_clamped
  elseif opt.multiplicative_integration == 4 then
    combiner = summation_w_gain
  end


  local n = 4 --number of layers per skippable block 
  local blocks =  math.floor(total_num_layers / n)
  if n * blocks ~= total_num_layers then
    print("ERROR: num_layers not divisible by num blocks")
  end
  print("Creating a stochastic depth RNN with " .. blocks .. " blocks of " .. n .. " layers")
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for b = 1, blocks do
    for L = 1,n do
      table.insert(inputs, nn.Identity()()) 
    end
  end

  function init(layer, mix)
    mix = mix or 1 --number between [0,1] that determines how much of the variance should be determined by this weight matrix 
    --if LSUV init is used, the value here will be overwritten by that procedure
    return layer:init('weight', nninit.normal, 0, math.sqrt(mix/rnn_size))
  end

  function normalization()
    if opt.layer_normalization == 2 then
        return nn.Normalize(2)
    elseif opt.layer_normalization ~= 0 then
        return nn.LayerNormalization(rnn_size)-- nil, nil, false)
    elseif opt.batch_normalization ~= 0 then
        return nn.BatchNormalization(rnn_size, 1e-5, 0.1, true)
    end
    return nn.Identity()
  end

  function idxToVector(idx)
    if embedding ~= nil then
      return embedding(idx)--nn.Tanh()(embedding(idx))
    else
      return OneHot(input_size)(idx)
    end
  end

  local skipWeight = 0.99 --the skip connection is multiplied by this scalar
  function rnn_layers(n, input_size_L, rnn_size, b)
    local x = nn.Identity()()
    local inputs = { [1] = x }
    local outputs = {}
    for L = 1, n do
      local prev_h = nn.Identity()()
      inputs[L+1] = prev_h

      if opt.recurrent_dropout > 0 then prev_h = nn.SharedDropout(opt.recurrent_dropout, "L" .. L ..  "b" .. b)(prev_h) end
      if opt.zoneout > 0 then prev_h = nn.Zoneout(opt.zoneout)(prev_h) end

      -- RNN tick
      local bias = opt.multiplicative_integration == 0 or opt.multiplicative_integration == 6

      local mix_i = 0.5
      local mix_h = 1-mix_i

      local corr = L == n and 1-(skipWeight^2) or 1 --compensate for skip


      local i2h = init(nn.Linear(input_size_L, rnn_size, bias), mix_i*corr)
          (x):annotate{name='i2h_'..(((b-1)*n)+L)}
    
      local h2h = init(nn.Linear(rnn_size, rnn_size, false), mix_h*corr)
          (prev_h):annotate{name='h2h_'..(((b-1)*n)+L)}

      if opt.activation_l1 ~= 0 then
          h2h = nn.L1Penalty(opt.l1)(h2h)
      end

      local activationFn = BELU --nn.BReLU

      for r=1, opt.recurrent_depth do
          print("Adding extra h2h matrix" .. r)
          h2h = init(nn.Linear(rnn_size, rnn_size))(h2h)       
          h2h = activationFn()(h2h)   
      end

      local next_h = combiner(rnn_size, L, i2h, h2h)

      next_h = activationFn()(normalization()(next_h))

      if opt.activation_clamp ~= 0 then
        next_h = nn.HardTanh(-opt.activation_clamp, opt.activation_clamp)(next_h)
      end

      if opt.activation_l2 ~= 0 then
        next_h = nn.L2Penalty(opt.activation_l2)(next_h)
      end

      table.insert(outputs, next_h)
      x = next_h
      if dropout > 0 and L < n then x = nn.Dropout(dropout)(x) end
    end
    table.insert(outputs, nn.Identity()(outputs[#outputs])) -- duplicate last output (we need this in the skipblocks)
    return nn.gModule(inputs, outputs)
  end

  function skip_rnn_layers(n, lstm)
    local x = nn.Identity()()
    local inputs = { [1] = x }
    local outputs = { }
    for L=1, n do 
      local copies = lstm and 2 or 1
      for c=1, copies do 
        local inp =  nn.Identity()()
        table.insert(inputs, inp)
        table.insert(outputs, inp)
      end
    end
    table.insert(outputs, x)
    return nn.gModule(inputs, outputs)
  end

  local x = idxToVector(inputs[1])
  local input_size_L = input_size
  if dropout > 0 and embedding then x = nn.Dropout(dropout)(x) end

  local lstm = false
  local outputs = {}
  for b = 1, blocks do

    local prev_hs = {}
    for L = 1, n do
      table.insert(prev_hs, inputs[((b-1)*n)+L+1])
    end

    local skip = skip_rnn_layers(n, lstm)
    local rnn_creator = lstm and oply_lstm_layers or rnn_layers
    local block = rnn_creator(n, input_size_L, rnn_size, b)

    local skipRate = 0.025

    print("skipRate", skipRate)
    local next_hs = table.pack(nn.StochasticSkip(block, skip, b, skipRate, false)({ x, unpack(prev_hs) }):split(n+1))
    x = nn.MulConstant(skipWeight, false)(x) --scale down the skip connection
    next_hs[#next_hs] = nn.CAddTable(true)({ next_hs[#next_hs], x })
    for L=1, n do
      table.insert(outputs, next_hs[L])
    end
    x = next_hs[#next_hs]
    if dropout > 0 then x = nn.Dropout(dropout)(x) end
    input_size_L = rnn_size

  end

  -- set up the decoder
  local top_h = x


  local proj = init(nn.Linear(rnn_size, vocab_size))(top_h)
  
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  return nn.gModule(inputs, outputs)
end

return SDRNN
