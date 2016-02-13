local SharedDropout, Parent = torch.class('nn.SharedDropout', 'nn.Dropout')

SharedDropout_noise = torch.CudaTensor()

function SharedDropout:__init(p,v1,inplace)
   Parent.__init(self, p, v1, inplace)
   self.noise = SharedDropout_noise
end

function SharedDropout:updateOutput(input)
   if self.inplace then 
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.train then
      -- SharedDropout_noise:resizeAs(input)
      -- self.noise:resizeAs(input)
      -- self.noise:bernoulli(1-self.p)
      -- if self.v2 then
      --    self.noise:div(1-self.p)
      -- end
      self.output:cmul(SharedDropout_noise)
   elseif not self.v2 then
      print("ERROR: v1 in SharedDropout NOT IMPLEMENTED")
      self.output:mul(1-self.p)
   end
   return self.output
end

function SharedDropout:updateGradInput(input, gradOutput)
   if self.train then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
      self.gradInput:cmul(SharedDropout_noise) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end
