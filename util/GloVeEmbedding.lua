
function split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function numkeys(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

local GloVeEmbedding, parent = torch.class('GloVeEmbedding', 'nn.LookupTable')--torch.class('GloVeEmbedding', 'nn.Module')

function GloVeEmbedding:__init(word2idx, embedding_file, embedding_size, data_dir)
    self.vocab_size = numkeys(word2idx)
    parent.__init(self, self.vocab_size, embedding_size)
    print("loading glove vectors")
    self.embedding_size = embedding_size
    local vocab_embedding_file = path.join(data_dir, "glove_" .. self.vocab_size .. ".t7")
    local loaded = {}
    if path.exists(vocab_embedding_file) then
        self.weight = torch.load(vocab_embedding_file)
    else
        local word_lower2idx = {}
        for word, idx in pairs(word2idx) do
            word_lower2idx[word:lower()] = idx
        end

        for line in io.lines(embedding_file) do
            local parts = split(line, " ")
            local word = parts[1]
            if word_lower2idx[word] then
                local idx = word_lower2idx[word]
                for i=2, #parts do
                    self.weight[idx][i-1] = tonumber(parts[i])
                end
                loaded[word] = true
            end
        end
        for word, idx in pairs(word2idx) do
            if not loaded[word:lower()] then
                print("Not loaded: " .. word:lower())
                for i=1, self.embedding_size do
                    self.weight[idx][i] = torch.normal(0, 0.01) --better way to do this?
                end
            end
        end
        torch.save(vocab_embedding_file, self.weight)
    end
    print("loaded glove vectors")
end
