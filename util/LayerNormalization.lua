
local LayerNormalization, parent = torch.class('LayerNormalization', 'nn.Sequential')

function LayerNormalization:__init(nOutput, bias, eps, affine)
   parent.__init(self)
   eps = eps or 1e-10
   affine = (affine == nil) and true or affine
   bias = bias or 0

   self:add(nn.ConcatTable()
               :add(nn.Identity())
               :add(nn.Sequential()
                       :add(nn.Mean(1, 1))
                       :add(nn.Replicate(nOutput,1,1))))
      :add(nn.CSubTable())
      :add(nn.Normalize(2, eps))
      :add(nn.MulConstant(torch.sqrt(nOutput)))

   if affine then
      local biasTransform = nn.Add(nOutput, false)
      biasTransform.bias:fill(bias)
      local gainTransform = nn.CMul(nOutput)
      gainTransform.weight:fill(1.)
      self:add(gainTransform)
      self:add(biasTransform)
   end
end

--[[
function nn.LayerNormalization(nOutput, bias, eps, affine, meancenter)
   local eps = eps or 1e-5
   if affine == nil then
      affine = true
   end
   if meancenter == nil then
      meancenter = true
   end

   local bias = bias or nil 

   local input = nn.Identity()()
   local input_center = input
   if meancenter then
       local mean = nn.Mean(2)(input)
       local mean_rep = nn.Replicate(nOutput,2)(mean) 
       input_center = nn.CSubTable()({input, mean_rep})
   end
   local std = nn.Sqrt()(nn.Mean(2)(nn.Square()(input_center)))
   local std_rep = nn.AddConstant(eps)(nn.Replicate(nOutput,2)(std))
   local output = nn.CDivTable()({input_center, std_rep})

   if affine then
      local biasTransform = nn.Add(nOutput, false)
      if bias ~= nil then
         biasTransform.bias:fill(bias)
      end
      local gainTransform = nn.CMul(nOutput)
      gainTransform.weight:fill(1.)
      output = biasTransform(gainTransform(output))
   end
   return nn.gModule({input},{output})
end
]]--
