

--[[
require 'models/CMulConstant'
function BELU(size)
    local s = nn.Sequential()
    s:add(nn.MulConstantSlices(torch.Tensor({ 1, -1 }):repeatTensor(size/2), false, 2))
    s:add(nn.ELU())
    s:add(nn.MulConstantSlices(torch.Tensor({ 1, -1 }):repeatTensor(size/2), false, 2))
    return s
end

return BELU 
--]]
local BELU , parent = torch.class('BELU','nn.ELU')

function BELU:__init()
  parent.__init(self)
end

function BELU:updateOutput(input)

  local s = input:size(2) / 2
  input[{{}, {1,s}}]:mul(-1)
  parent.updateOutput(self, input)
  input[{{}, {1,s}}]:mul(-1)
  
  s = self.output:size(2) / 2
  self.output2 = self.output2 or self.output.new()
  self.output2:resizeAs(self.output):copy(self.output)
  self.output2[{{}, {1,s}}]:mul(-1)
  
  return self.output2
end

function BELU:updateGradInput(input, gradOutput)
  local s = input:size(2) / 2
  input[{{}, {1,s}}]:mul(-1)
  --gradOutput = gradOutput:clone()
  --gradOutput[{{}, {1,s}}]:mul(-1)
  parent.updateGradInput(self, input, gradOutput)
  input[{{}, {1,s}}]:mul(-1)
  --gradOutput[{{}, {1,s}}]:mul(-1)

  return self.gradInput
end

function BELU:__tostring__()
   return torch.type(self) .. string.format('()')
end

