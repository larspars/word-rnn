
local LookupTableFixed, parent = torch.class('LookupTableFixed', 'nn.LookupTable')

function LookupTableFixed:__init(nIndex, nOutput, paddingValue, maxNorm, normType)
    parent.__init(self, nIndex, nOutput, paddingValue, maxNorm, normType)
end

function LookupTableFixed:accGradParameters(input, gradOutput, scale)
end

function LookupTableFixed:parameters()
    return {}, {}
end

