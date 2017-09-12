require 'nn'
require 'nngraph'

local Zoneout, Parent = torch.class('nn.Zoneout', 'nn.Module')


function Zoneout:__init(p)
   Parent.__init(self)
   self.p = p

   local input = nn.Identity()()  
   local prevOutput = nn.Identity()()
   local sub = nn.CSubTable(true)({input, prevOutput})
   local dropout = nn.Dropout(self.p, true, true)(sub)
   local output = nn.CAddTable(true)({dropout, prevOutput})
   self.gmodule = nn.gModule({input, prevOutput},{output})
   self.prevOutput = torch.Tensor()
end


function Zoneout:updateOutput(input)
   if self.output:nElement() ~= input:nElement() then
      self.output:resizeAs(input):zero()
   end
   if self.prevOutput:nElement() ~= input:nElement() then
      self.prevOutput:resizeAs(input):zero()
   end
   
   self.prevOutput:copy(self.output)
   if self.train then
      self.output:copy(self.gmodule:forward({input, self.output}))
   else
      self.output:copy(input):mul(1-self.p):add(self.p, self.prevOutput)
   end
   return self.output
end


function Zoneout:updateGradInput(input, gradOutput)
   local grads = self.gmodule:backward({input, self.prevOutput}, gradOutput)
   self.gradInput = grads[1]
   return self.gradInput
end

function Zoneout:training()
   Parent.training(self)
   self.gmodule:training()
end

function Zoneout:evaluate()
   Parent.evaluate(self)
   self.gmodule:evaluate()
end




