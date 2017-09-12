local nninit = require 'nninit'

local function calcGain(gain)
  -- Return gain if a number already
  if type(gain) == 'number' then
    return gain
  end

  -- Extract gain string if table
  local args
  if type(gain) == 'table' then
    args = gain
    gain = gain[1]
  end

  -- Process gain strings with optional args
  if gain == 'linear' or gain == 'sigmoid' then
    return 1
  elseif gain == 'relu' then
    return math.sqrt(2)
  elseif gain == 'lrelu' then
    return math.sqrt(2 / (1 + math.pow(args.leakiness, 2)))
  end

  -- Return 1 by default
  return 1
end

--[[
nninit.orthogonal = function(module, tensor, options)
  local sizes = tensor:size()
  if #sizes < 2 then
    error("nninit.orthogonal only supports tensors with 2 or more dimensions")
  end

  -- Calculate "fan in" and "fan out" for arbitrary tensors based on module conventions
  local fanIn = sizes[2]
  local fanOut = sizes[1]
  for d = 3, #sizes do
    fanIn = fanIn * sizes[d]
    fanOut = fanOut * sizes[d]
  end

  options = options or {}
  gain = calcGain(options.gain)

  print("orthogonalizing, gain " .. gain .. " " .. fanOut .. "x" .. fanIn)
  -- Construct random matrix
  local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)

  local W, R
  if fanOut == fanIn then
    W,R = torch.qr(randMat)
  elseif fanOut > fanIn then
    local Q,R = torch.qr(torch.Tensor(fanOut, fanOut):normal(0, 1))
    local Qsub = Q:narrow(2, 1, fanIn)
    W = torch.Tensor()
    W:resizeAs(Qsub):copy(Qsub)
  else
    local Q,R = torch.qr(torch.Tensor(fanIn, fanIn):normal(0, 1))
    local Qsub = Q:narrow(2, 1, fanIn)
    W = torch.Tensor()
    W:resizeAs(Qsub):copy(Qsub)
  end
  -- Resize
  W:resize(tensor:size())
  -- Multiply by gain
  W:mul(gain)

  tensor:copy(W)
  return module
end
]]--
return nninit
