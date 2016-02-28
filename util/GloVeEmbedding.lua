
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

local GloVeEmbedding, parent = torch.class('GloVeEmbedding', 'nn.LookupTable')

function GloVeEmbedding:__init(word2idx, embedding_size, data_dir)
    -- you need glove embeddings in the directory ./util/glove/
    -- download them from http://nlp.stanford.edu/projects/glove/
    local embedding_file = 'util/glove/vectors.6B.200d.txt' 
    local file_embedding_size = 200
    self.vocab_size = numkeys(word2idx)
    self.word2idx = word2idx
    parent.__init(self, self.vocab_size, embedding_size)
    print("loading glove vectors")
    self.embedding_size = embedding_size
    --load glove vectors as a tensor from a .t7 file if it exists, otherwise generate that .t7 file
    local vocab_embedding_file = path.join(data_dir, "glove_" .. self.vocab_size .. "x" .. embedding_size .. ".t7")
    --vocab_embedding_file = "lstm-glove-w.t7"
    --(if you want to load a different word vector here, just swap the vocab_embedding_file file name with something else)
    if path.exists(vocab_embedding_file) then
        self.weight = torch.load(vocab_embedding_file):clone()
    else
        w = self:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
        if file_embedding_size ~= embedding_size then
            w = torch.mm(w, torch.rand(file_embedding_size, embedding_size))
        end
        self.weight = w:contiguous()
        torch.save(vocab_embedding_file, self.weight)
    end
    print("loaded glove vectors")
end

function GloVeEmbedding:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
    local word_lower2idx = {}
    local loaded = {}
    local weight = torch.Tensor(self.vocab_size, file_embedding_size)
    for word, idx in pairs(word2idx) do
        word_lower2idx[word:lower()] = idx
    end

    for line in io.lines(embedding_file) do
        local parts = split(line, " ")
        local word = parts[1]
        if word_lower2idx[word] then
            local idx = word_lower2idx[word]
            for i=2, #parts do
                weight[idx][i-1] = tonumber(parts[i])
            end
            loaded[word] = true
        end
    end
    for word, idx in pairs(word2idx) do
        if not loaded[word:lower()] then
            print("Not loaded: " .. word:lower())
            for i=1, file_embedding_size do
                weight[idx][i] = torch.normal(0, 0.9) --better way to do this?
            end
        end
    end
    return weight
end

function GloVeEmbedding:updateOutput(input)
    return parent.updateOutput(self, input:contiguous())
end

function GloVeEmbedding:accGradParameters(input, gradOutput, scale)
    return parent.accGradParameters(self, input:contiguous(), gradOutput:contiguous(), scale)
end


local GloVeEmbeddingFixed, parent = torch.class('GloVeEmbeddingFixed', 'GloVeEmbedding')

function GloVeEmbeddingFixed:accGradParameters(input, gradOutput, scale)
    return nil
end
function GloVeEmbeddingFixed:parameters()
    return {}, {}
end

--[[
function GloVeEmbeddingProject:__init(word2idx, embedding_size, data_dir)
    local embedding_file = 'util/glove/vectors.6B.200d.txt'
    local file_embedding_size = 200
    self.vocab_size = numkeys(word2idx)
    parent.__init(self, self.vocab_size, embedding_size)
    print("loading glove vectors")
    self.embedding_size = embedding_size
    local vocab_embedding_file = path.join(data_dir, "glove_" .. self.vocab_size .. "x" .. embedding_size .. ".t7")
    --print("loading pretrained word vectors")
    --vocab_embedding_file = "glove_embeddings_pretrained1862x200-4sitelow-descs.t7"
    --vocab_embedding_file = "glove_embeddings_pretrained1704x200-5site-title.t7"
    if path.exists(vocab_embedding_file) then
        self.weight = torch.load(vocab_embedding_file):clone()
    else
        w = self:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
        if file_embedding_size ~= embedding_size then
            w = torch.mm(w, torch.rand(file_embedding_size, embedding_size))
        end
        self.weight = w:contiguous()
        torch.save(vocab_embedding_file, self.weight)
    end
    print("self.weight size")
    print(self.weight:size())
    print("loaded glove vectors")
end



local GloVeEmbeddingFixed, parent = torch.class('GloVeEmbeddingFixed', 'nn.Module')
function GloVeEmbeddingFixed:__init(word2idx, embedding_file, embedding_size, data_dir)
    self.vocab_size = numkeys(word2idx)
    parent.__init(self, self.vocab_size, embedding_size)
    print("loading glove vectors")
    self.embedding_size = embedding_size
    local vocab_embedding_file = path.join(data_dir, "glove_" .. self.vocab_size .. "x" .. embedding_size .. ".t7")
    print("loading pretrained word vectors")
    vocab_embedding_file = "glove_embeddings_pretrained1704x200.t7" --pretrained vectors
    local loaded = {}
    if path.exists(vocab_embedding_file) then
        self.weight = torch.load(vocab_embedding_file)
    else
        self.weight = torch.Tensor(self.vocab_size, embedding_size)
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
                    self.weight[idx][i] = torch.normal(0, 0.9) --better way to do this?
                end
            end
        end
        torch.save(vocab_embedding_file, self.weight)
    end
    print("loaded glove vectors")
end

function GloVeEmbeddingFixed:updateOutput(input)
  print("GloVeEmbeddingFixed input size: " .. input:size())
  self.output:resize(input:size(1), self.embedding_size):zero()
  local longInput = input:long()
  self.output:copy(self.weight:index(1, longInput))
  return self.output
end
]]--

