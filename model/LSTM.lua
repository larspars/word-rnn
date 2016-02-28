_ = require 'underscore'
local nninit = require 'nninit'

local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout, recurrent_dropout, embedding, num_fixed)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local fixeds = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      if embedding ~= nil then
        input_size_L = 200
        local embedded = embedding(inputs[1])
        x = nn.Tanh()(embedded)
      else
        x = OneHot(input_size)(inputs[1])
        input_size_L = input_size
      end
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end


    if recurrent_dropout > 0 then prev_h = nn.SharedDropout(dropout)(prev_h) end

    local i2h = nil
    local h2h = nil
    if L <= num_fixed then
	i2h = nn.LinearFixed(input_size_L, 4 * rnn_size, "lstm-l"..L.."-i2h-w.t7", "lstm-l"..L.."-i2h-b.t7")(x):annotate{name='i2h_'..L}
    	h2h = nn.LinearFixed(rnn_size, 4 * rnn_size, "lstm-l"..L.."-h2h-w.t7", "lstm-l"..L.."-h2h-b.t7")(prev_h):annotate{name='h2h_'..L}
        fixeds[#fixeds+1] = i2h.data.module
        fixeds[#fixeds+1] = h2h.data.module
    else 
    	i2h = nn.Linear(input_size_L, 4 * rnn_size):init('weight', nninit.uniform, -0.08, 0.08)(x):annotate{name='i2h_'..L}
    	h2h = nn.Linear(rnn_size, 4 * rnn_size):init('weight', nninit.uniform, -0.08, 0.08)(prev_h):annotate{name='h2h_'..L}
    end
    
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size):init('weight', nninit.uniform, -0.08, 0.08)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  local module = nn.gModule(inputs, outputs)
  function module.parametersNoGrad()
    --print(_.map(fixeds, function(fixed) return fixed:parametersNoGrad() end))
    return _.flatten(_.map(fixeds, function(fixed) return fixed:parametersNoGrad() end))
  end
  return module
end

function flatten(list)
  if type(list) ~= "table" then return {list} end
  local flat_list = {}
  for _, elem in ipairs(list) do
    for _, val in ipairs(flatten(elem)) do
      flat_list[#flat_list + 1] = val
    end
  end
  return flat_list
end

return LSTM

