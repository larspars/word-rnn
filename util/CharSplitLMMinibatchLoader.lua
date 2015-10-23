local rex = require 'rex_pcre'

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

    self.word_level = word_level
    local input_file = path.join(data_dir,  'input.txt')
    local vocab_file = path.join(data_dir, word_level and 'vocab_w' .. threshold .. '.t7' or 'vocab.t7')
    local tensor_file = path.join(data_dir, word_level and 'data_w' .. threshold .. '.t7' or 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
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

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

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

-- *** STATIC method ***
function CharSplitLMMinibatchLoader.text_to_tensor(word_level, threshold, in_textfile, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()
    print('loading text file...')
    local f = torch.DiskFile(in_textfile)
    local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
    f:close()

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    print('word occurence threshold is ' .. threshold)
    -- record all characters to a set
    local unordered = {}
    --rawdata = re.sub('([%s])' % (re.escape(string.punctuation)+"1234567890"), r" \1 ", rawdata)
    local numtokens = 0    
    for token in CharSplitLMMinibatchLoader.tokens(rawdata, word_level) do
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
    end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = word_level and torch.ShortTensor(numtokens) or torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    if word_level then
        local i = 1
        for token in CharSplitLMMinibatchLoader.tokens(rawdata, word_level) do
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

function CharSplitLMMinibatchLoader.tokens(rawstr, word_level)
    if word_level then
        --local str, _, _ = rex.gsub(rawstr, '[[:punct:][:digit:]]', ' %0 ')
        --str, _, _ = rex.gsub(str, '\\n', ' RN ')
        --return rex.split(str, "\\s+")
        return word_iter(rawstr)
    else
        return rawstr:gmatch'.'
    end
end

function word_iter(str)
    local n = str:len()
    local punctdigit = rex.new('[[:punct:][:digit:]]')
    local newline = rex.new('\\n')
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

return CharSplitLMMinibatchLoader











