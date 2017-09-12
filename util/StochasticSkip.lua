require 'nn'
require 'cudnn'
require 'cunn'

local StochasticSkip, parent = torch.class('nn.StochasticSkip', 'nn.Container')

function StochasticSkip:__init(net, skip, seed, skipRate, increaseSkipRate)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.gen = torch.Generator()-- we can keep layers in sync across timesteps by using the same seed
    torch.manualSeed(self.gen, seed or 123)
    self.skipRate = skipRate
    self.increaseSkipRate = increaseSkipRate or false
    self.skipRateTarget = 0.2
    self.initialized = false
    self.net = net
    self.skip = skip
    self.t = -1
    self.modules = {self.net, self.skip}
end

function StochasticSkip:updateOutput(input)
    
    if self.train then
        self.t = self.t + 1
        if self.increaseSkipRate and self.t ~= 0 and self.t % 783 == 0 and self.skipRate < self.skipRateTarget then
            self.skipRate = self.skipRate + self.skipRateTarget/30
            print("increased skip rate to ", self.skipRate)
        end
    end
    --for now, assume input and output to be a table (not a tensor)
    --self.output:resizeAs(skip_forward):copy(skip_forward)
    
    --self.gate = torch.uniform(self.gen) > self.skipRate
    self.gate = torch.uniform() > self.skipRate
    if self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output = self.net:forward(input)
      else
        self.output = self.skip:forward(input)
      end
    else
        --do we need to clone output vectors from the graph?
        --[[
      local fwd = self.net:forward(input)
      self.output = {}
      for i = 1, #fwd do
        self.output[i] = fwd[i]:clone():mul(1-self.skipRate)
      end
        ]]--

      self.output = self.net:forward(input)
      --why would we scale activations? 
      --each block has same expected activation magnitude, so scaling is wrong here
      --the recurrent setting exacerbates this error
      --for i = 1, #self.output do
      --  self.output[i] = self.output[i]:clone():mul(1-self.skipRate)
      --end
    end
    return self.output
end

function StochasticSkip:updateGradInput(input, gradOutput)
 --[[
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.gate then
      self.gradInput:add(self.net:updateGradInput(input, gradOutput))
   end
]]--
    if self.gate then
        self.gradInput = self.net:updateGradInput(input, gradOutput)
    else
        self.gradInput = self.skip:updateGradInput(input, gradOutput)
    end
    return self.gradInput
end

function StochasticSkip:accGradParameters(input, gradOutput, scale)
    if self.gate then
        self.gradInput = self.net:accGradParameters(input, gradOutput, scale)
    else
        self.gradInput = self.skip:accGradParameters(input, gradOutput, scale)
    end
end


