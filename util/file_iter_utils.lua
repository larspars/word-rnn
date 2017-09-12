local rex = require 'rex_pcre'

function PeekableIterator(iter)
    local t = {
        __call = function()
            return iter()
        end,
        pop = function(self)
            local val = self._peeked
            self._peeked = nil
            return val
        end, 
        peek = function(self)
            if self._peeked == nil then 
                self._peeked = self()
            end
            return self._peeked
        end
    }
    return setmetatable(t, t)
end

function sentence_iter(str)
    local iterator = word_iter(str)
    local done = false
    return function()
        
        local s = {}
        local word = nil
        if done then
            return nil
        end
        local i = 0
        for word in iterator do
            i = i + 1
            if word == '\n' then 
                break 
            end
            if word == nil then
                done = true
                return s
            end
            s[#s+1] = word
        end
        if i == 0 then
            done = true
        end
        return s
    end
end

local punctdigit = rex.new('[[:punct:][:digit:]]')
local newline = rex.new('\\n')
local whitespace = rex.new('[ \\t]') --dont match newlines

function word_iter(file) --assumes pre-tokenization, splits on whitespace
    file:quiet()
    local line = file:readString('*l')
    local word_iter = line:gmatch("%S+")
    line_count = 1
    return function()
        local w = word_iter()
        if w ~= nil then
            return w
        else
            line = file:readString('*l')
            if file:hasError() then
                return nil
            end
            line_count = line_count + 1
            word_iter = line:gmatch("%S+")
            return '\n' -- nextchar()
        end
    end
end


function word_iter_old(file)
    file:quiet()
    local line = file:readString('*l')
    local char_iter = line:gmatch'.'
    line_count = 1
    local function nextchar()
        local c = char_iter()
        if c ~= nil then
            print("c'" .. c .. "'")
            return c
        else
            line = file:readString('*l')
            print(line)
            if file:hasError() then
                return nil
            end
            line_count = line_count + 1
            char_iter = line:gmatch'.'
            return '\n' -- nextchar()
        end
    end
    local c = nextchar()
    return function()
        if c == nil then return nil end
        while rex.count(c, whitespace) > 0 do
            c = nextchar()
            if c == nil then return nil end
        end
        if rex.count(c, punctdigit) > 0 then
            local ret = c
            c = nextchar()
            return ret
        end
        if rex.count(c, newline) > 0 then
            c = nextchar()
            return '\n'
        end
        local word = ''
        repeat
            word = word .. c
            c = nextchar()
            if c == nil then return word end
        until rex.count(c, whitespace) > 0 or rex.count(c, punctdigit) > 0 or rex.count(c, newline) > 0
        
        print("w'" .. word .. "'")
        return word
    end
end
