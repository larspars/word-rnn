function softplus(x) 
    local lt20 = torch.lt(x, 20):double()
    local vals_gt20 = torch.cmul(x, torch.ge(x, 20):double())
    return torch.log(1 + torch.exp(torch.cmul(x, lt20))):cmul(lt20):add(vals_gt20) 
end

local function lsuvInit(net, vocab_size)
    local function extractLayerModules(net, layer_idx)
        local result = {}
        for _, node in ipairs(net.forwardnodes) do
            if torch.type(node.data.module) == 'nn.StochasticSkip' then
                local modules = extractLayerModules(node.data.module.net, layer_idx)
                for k,v in ipairs(modules) do
                    table.insert(result, v)
                end
            else
                local name = node.data.annotations.name
                if name == "i2h_" .. layer_idx or name == "h2h_" .. layer_idx then
                    table.insert(result, node.data.module)
                end
                
            end
        end
        return result
    end
    net:evaluate()
    local x = { gpu_prepare(torch.Tensor(opt.batch_size):random(1, vocab_size)) }
    local isLstm = opt.model == 'lstm' or opt.model == 'blstm' or opt.model == 'blstm2'
    for l=1, opt.num_layers do
        local state = gpu_prepare(torch.Tensor(opt.batch_size, opt.rnn_size)):normal(0, 1)
        table.insert(x, state)
        if isLstm then
            local state = gpu_prepare(torch.Tensor(opt.batch_size, opt.rnn_size)):normal(0, 1)
            table.insert(x, state)
        end
    end
    local lastStd = 0 
    local start = isLstm and 2 or 1
    local N = isLstm and opt.num_layers*2 or opt.num_layers
    local step = isLstm and 2 or 1
    for l=1, opt.num_layers do
        local modules = extractLayerModules(net, l)
        for k,module in pairs(modules)  do
            local std = module.weight:std()
            if std > 0 and lastStd ~= 0 then
                module.weight:div(module.weight:std()):mul(lastStd) --init std to what last layer ended up at
            end
        end
        local state = net:forward(x)
        local stateIdx = isLstm and l*2 or l
        local var = state[stateIdx]:var()
        print("layer ".. l .. " output variance was", var)
        iters = 0
        for k,v in pairs(modules) do
            print("std: " .. k, v.weight:std())
        end
        while math.abs(var - 1) > 0.001 and iters < 40 do
            for k,module in pairs(modules) do
                local adjust = 1 --l % 4 == 0 and softplus(torch.Tensor({var}))[1]/var or 1
                module.weight:div(math.sqrt(var * adjust))
                print(string.format("layer %d, [w: %d] weight now has std", l, k), module.weight:std(), "output variance was", var)
            end
            local state = net:forward(x)
            var = state[stateIdx]:var()
            iters = iters + 1
        end
        if #modules > 0 then
            lastStd = modules[1].weight:std()
            for k,v in pairs(modules) do
                print("std: " .. k, v.weight:std())
            end
        end
    end
    net:training()
end

return lsuvInit
