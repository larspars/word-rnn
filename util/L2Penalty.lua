local L2Penalty, parent = torch.class('nn.L2Penalty','nn.Module')

--This module penalizes the L2 size of the activations

function L2Penalty:__init(l2weight)
    parent.__init(self)
    self.l2weight = l2weight 
end
    
function L2Penalty:updateOutput(input)
    self.output = input 
    return self.output 
end

function L2Penalty:updateGradInput(input, gradOutput)    
    self.gradInput:resizeAs(input):copy(gradOutput):add(self.l2weight, input)
    return self.gradInput 
end
