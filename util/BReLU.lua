local BReLU , parent = torch.class('nn.BReLU','nn.ReLU')

function BReLU:__init()
  parent.__init(self)
end

function BReLU:updateOutput(input)

  local s = input:size(2) / 2
  input[{{}, {1,s}}]:mul(-1)
  parent.updateOutput(self, input)
  input[{{}, {1,s}}]:mul(-1)
  
  s = self.output:size(2) / 2
  self.output[{{}, {1,s}}]:mul(-1)
  
  return self.output
end

function BReLU:updateGradInput(input, gradOutput)
  
  local s = input:size(2) / 2
  input[{{}, {1,s}}]:mul(-1)
  parent.updateGradInput(self, input, gradOutput)
  input[{{}, {1,s}}]:mul(-1)
  
  return self.gradInput
end

function BReLU:__tostring__()
   return torch.type(self) .. string.format('()')
end

