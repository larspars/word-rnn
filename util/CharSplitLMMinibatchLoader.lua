local rex = require 'rex_pcre'

require 'util/file_iter_utils'

-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local CharSplitLMMinibatchLoader = {}
CharSplitLMMinibatchLoader.__index = CharSplitLMMinibatchLoader

function CharSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, word_level, threshold)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    if not word_level then 
        threshold = 0 
    end
    local self = {}
    setmetatable(self, CharSplitLMMinibatchLoader)

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.word_level = word_level
    local input_file = path.join(data_dir,  'input.txt')
    local vocab_file = path.join(data_dir, word_level and 'vocab_w' .. threshold .. '.t7' or 'vocab.t7')
    local tensor_file = path.join(data_dir, word_level and 'data_w' .. threshold .. '.t7' or 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) and path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        CharSplitLMMinibatchLoader.text_to_tensor(word_level, threshold, input_file, vocab_file, tensor_file)
    end

    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)


    function round(n) return math.floor(0.5 + n) end
    local split1 = round(data:size(1)*split_fractions[1])
    local split2 = split1 + round(data:size(1)*split_fractions[2])
    self.fullData = {
        --TODO: do we need to clone() here, to ensure that splits stay seperate? check that we cant expand a subview into a non-sub-view
        data[{{1,      split1}}]:clone(),
        data[{{split1 + 1, split2}}]:clone(),
        data[{{split2 + 1, data:size(1)}}]:clone()
    }
    print("valid starts at:", split1+1, "test starts at:", split2+1)
    self.x_batches = {}
    self.nbatches = {}
    self.y_batches = {}
    
    self:set_data(1, self:evenCrop(self.fullData[1], opt.random_crops == 1))
    self:set_data(2, self:evenCrop(self.fullData[2], false))
    self:set_data(3, self:evenCrop(self.fullData[3], false))


    --[[
    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end
    --]]

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    --[[
    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)
    --]]

    -- lets try to be helpful here
    if self.nbatches[1] < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

    if split_fractions[3] == 0 then print("WARNING: Not setting aside a test set") end

    -- divide data to train/val and allocate rest to test
    self.ntrain = self.nbatches[1] --math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches[2] --math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches[3] --self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}


    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function CharSplitLMMinibatchLoader:evenCrop(data, randomize)
    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (self.batch_size * self.seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        local evenLen = self.batch_size * self.seq_length * math.floor(len / (self.batch_size * self.seq_length))
        local overflow = len - evenLen
        local offset = 0
        if randomize then
            offset = math.floor(torch.uniform(0, overflow))
            self.overflow = overflow
        end
        return data:sub(1+offset, evenLen+offset)
    end
    return data
end

    
function CharSplitLMMinibatchLoader:set_data(split_index, data)
    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    local v = data:view(self.batch_size, -1)
    self.x_batches[split_index] = v:split(self.seq_length, 2)  -- #rows = #batches
    self.nbatches[split_index] = #self.x_batches[split_index]
    self.y_batches[split_index] = ydata:view(self.batch_size, -1):split(self.seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches[split_index] == #self.y_batches[split_index])
    collectgarbage()
end


function CharSplitLMMinibatchLoader.create_vocab(word_level, threshold, in_textfile, out_vocabfile)
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    print('word occurence threshold is ' .. threshold)
    -- record all characters to a set
    local unordered = {}
    --rawdata = re.sub('([%s])' % (re.escape(string.punctuation)+"1234567890"), r" \1 ", rawdata)
    local numtokens = 0    
    local j = 0
    for token in CharSplitLMMinibatchLoader.tokens(in_textfile, word_level) do
        j = j + 1
        if j % 500000 == 0 then print("token " .. j) end
        if not unordered[token] then 
            unordered[token] = 1 
        else
            unordered[token] = unordered[token] + 1 
        end
        numtokens = numtokens + 1
    end
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for token, count in pairs(unordered) do 
        if count > threshold then
            ordered[#ordered + 1] = token 
        end
    end
    if word_level then
        ordered[#ordered + 1] = "UNK" --represents unknown words
        ordered[#ordered + 1] = "<EOS>" --end of sentence
        ordered[#ordered + 1] = "<NULL>" --all zero embedding, to be ignored.
    end
    --table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    if not word_level then 
        error("Char level seq2seq not supported")
    end

    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    return vocab_mapping
end

function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

--[[
function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end
--]]
function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        if split_index == 1 and opt.random_crops == 1 then
            print("Choosing a new random crop of the training data")
            self:set_data(split_index, self:evenCrop(self.fullData[split_index], true))
        end
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    return self.x_batches[split_index][ix], self.y_batches[split_index][ix]
end

local rawdata = nil --used when word_level=0.

--[[
function CharSplitLMMinibatchLoader.text_to_tensor(word_level, numtokens, vocab_mapping, in_textfile, out_tensorfile)
    print('putting data into tensor...')
    local data = word_level and torch.IntTensor(numtokens) or torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    if word_level then
        local i = 1
        for token in CharSplitLMMinibatchLoader.tokens(in_textfile, word_level) do
            data[i] = vocab_mapping[token] or vocab_mapping["UNK"]
            i = i + 1
        end
    else
		for i=1, #rawdata do
			data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
		end
    end

    -- save output preprocessed files
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end
--]]

function CharSplitLMMinibatchLoader.text_to_tensor(word_level, threshold, in_textfile, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    print('word occurence threshold is ' .. threshold)
    -- record all characters to a set
    local unordered = {}
    --rawdata = re.sub('([%s])' % (re.escape(string.punctuation)+"1234567890"), r" \1 ", rawdata)
    local numtokens = 0    
    collectgarbage()
    for token in CharSplitLMMinibatchLoader.tokens(in_textfile, word_level) do --rawdata, word_level) do     
        if not unordered[token] and numtokens > 30000000 then
            --ignore it, or we'll blow up memory wise (would likely be excluded anyway)
        elseif not unordered[token] then 
            unordered[token] = 1 
        else
            unordered[token] = unordered[token] + 1 
        end
        numtokens = numtokens + 1
    end
    print('num tokens:', numtokens)
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for token, count in pairs(unordered) do 
        if count > threshold then
            ordered[#ordered + 1] = token 
        end
    end
    unordered = nil
    collectgarbage()
    if word_level then
        ordered[#ordered + 1] = "UNK" --represents unknown words
        ordered[#ordered + 1] = "<EOS>" --end of sentence (for compat with seq2seq)
        ordered[#ordered + 1] = "<NULL>" --all zero embedding, to be ignored. (for compat with seq2seq)
    end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('vocabulary size:', #ordered)
    print('putting data into tensor...')
    local data = word_level and torch.IntTensor(numtokens) or torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    if word_level then
        local i = 1
        for token in CharSplitLMMinibatchLoader.tokens(in_textfile, word_level) do
            data[i] = vocab_mapping[token] or vocab_mapping["UNK"]
            i = i + 1
        end
    else
		for i=1, #rawdata do
			data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
		end
    end

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

function CharSplitLMMinibatchLoader.tokens(filename, word_level)
    if word_level then
        local file = torch.DiskFile(filename)
        return word_iter(file)
    else
        print('loading text file...')
        local f = torch.DiskFile(filename)
        rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
        f:close()
        return rawdata:gmatch'.'
    end
end


--function word_iter(str)
--    local n = str:len()
--    local punctdigit = rex.new('[[:punct:][:digit:]]')
--[[    local newline = rex.new('\\n')
    local whitespace = rex.new('[ \\t]') --dont match newlines
    local char_iter = str:gmatch'.'
    local c = char_iter()
    return function()
        if c == nil then return nil end
        while rex.count(c, whitespace) > 0 do
            c = char_iter()
            if c == nil then return nil end
        end
        if rex.count(c, punctdigit) > 0 then
            local ret = c
            c = char_iter()
            return ret
        end
        if rex.count(c, newline) > 0 then
            c = char_iter()
            return '\n'
        end
        local word = ''
        repeat
            word = word .. c
            c = char_iter()
            if c == nil then return word end
        until rex.count(c, whitespace) > 0 or rex.count(c, punctdigit) > 0 or rex.count(c, newline) > 0
        
        return word
    end
end
--]]
return CharSplitLMMinibatchLoader











