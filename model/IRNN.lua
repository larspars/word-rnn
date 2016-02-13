--require 'util.ELU'
--require 'util.LinearFixed'
--require 'util.L2Linear'
--require 'util.EigenvalueL2Linear'

local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout, embedding)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local h2hs = {}
  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      if embedding ~= nil then
        input_size_L = 200
        local embedded = embedding(inputs[1])
print("**********EMBEDDING**********")
        x = nn.Tanh()(embedded)
      else
print("############OneHot###########")
        x = OneHot(input_size)(inputs[1])
        input_size_L = input_size
      end
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local init = 0.9
    local i2h = nil
    if input_size_L == rnn_size then
      i2h = nn.Linear(input_size_L, rnn_size)(x) 
      --i2h = nn.EigenvalueL2Linear(input_size_L, rnn_size, "i-h"..L)(x)
      i2h.data.module.weight:eye(rnn_size):mul(init)
    else
      i2h = nn.Linear(input_size_L, rnn_size)(x)
    end
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h) 
    --local h2h = nn.L2Linear(rnn_size, rnn_size, 0.0015)(prev_h) 
    --local h2h = nn.EigenvalueL2Linear(rnn_size, rnn_size, "h-h"..L)(prev_h)

    h2h.data.module.weight:eye(rnn_size):mul(init)
    local next_h = nn.ELU()(nn.CAddTable(){i2h, h2h})
    --local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})
    --local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
    table.insert(h2hs, h2h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs), h2hs
end

return RNN
