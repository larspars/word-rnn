local SharedDropout, Parent = torch.class('nn.SharedDropout', 'nn.Dropout')

SharedDropout_noise = {} 
SharedDropout_ps = {} 

function SharedDropout:__init(p, id, inplace)
   Parent.__init(self, p, false, inplace)
   SharedDropout.init(id, p)
   -- self.noise = SharedDropout_noise[id]
   self.id = id
end

function SharedDropout:updateOutput(input)
   if self.inplace then 
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.train then
      if not SharedDropout_noise[self.id]:isSameSizeAs(input) then
         SharedDropout_noise[self.id]:resizeAs(input)
         SharedDropout.reset_mask(SharedDropout_noise[self.id], SharedDropout_ps[self.id])
      end
      -- SharedDropout_noise:resizeAs(input)
      -- self.noise:bernoulli(1-self.p)
      -- if self.v2 then
      --    self.noise:div(1-self.p)
      -- end
      self.output:cmul(SharedDropout_noise[self.id])
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
      self.gradInput:cmul(SharedDropout_noise[self.id]) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function SharedDropout.init(id, p)
   if SharedDropout_noise[id] == nil then
      SharedDropout_noise[id] = torch.CudaTensor()
      SharedDropout_ps[id] = p
   end
end

function SharedDropout.reset() 
    for k,v in pairs(SharedDropout_noise) do
        if v:dim() > 0 then
            SharedDropout.reset_mask(v, SharedDropout_ps[k])
        end
    end
end

function SharedDropout.reset_mask(mask, p) 
    mask:bernoulli(1 - p)
    mask:div(1 - p)
end
