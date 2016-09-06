
-- LM -- Input/Output Utils -- 11/8/16 --



-- function reading filelist
function readFilelist(filelist)
  
  -- check if the file exists
  if not paths.filep(filelist) then  
    error('File ' .. filelist .. ' does not exist!')
  end
  
  local file = io.open(filelist, "r")
  io.input(file)
  
  -- if view -> skip the info line
  if settings.inputView > 0 then
    local info = io.read()
  end  
  
  -- load lines to process
  local files = {}
  for line in io.lines() do
    line = string.gsub(line, "%s+", "")
    table.insert(files, line)
  end
  
  io.close(file)
  
  return files
  
end



-- function reading DNN inputs
function readInputs(file, start, stop)
  
  -- switch to correct reading
  if settings.inputType == "htk" then
    return readHTK(file, start, stop)
  else
    error('InputType: not supported')
  end
  
end



-- function reading DNN inputs - HTK format
function readHTK(file, start, stop) 
  
  -- check if the file exists
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!')
  end
  
  -- read file
  local f = torch.DiskFile(file, 'r')
  f:binary()
  
  -- read HTK header
  local nSamples = f:readInt()
  local sampPeriod = f:readInt()
  local sampSize = f:readShort()
  local parmKind = f:readShort()

  -- prepare to read parts
  if start and stop then
    f:seek(f:position() + ((start-1) * settings.inputSize * 4))
    nSamples = stop - start + 1
  end
    
  -- ntx4 alignment fix
  if settings.dnnAlign == 1 then
    nSamples = nSamples - 1
  end
  
  -- read features
  local data = f:readFloat(nSamples * sampSize / 4)  
  f:close()

  -- create tensor from input data
  local fvec = torch.Tensor(data, 1, torch.LongStorage{nSamples, sampSize / 4})
  
  return nSamples, sampPeriod, sampSize, parmKind, fvec
  
end



-- function reading DNN inputs & outputs - view 
function readView(file, prepareRefs)
  
  if settings.inputView == 1 then
    return readViewV1(file, prepareRefs)
  elseif settings.inputView == 2 then 
    return readViewV2(file, prepareRefs)
  else
    error('ViewType: not supported')
  end
  
end
  
  
  
-- function reading DNN inputs & outputs - view (SAD version - v1)
function readViewV1(file, prepareRefs)
  
  -- only 2 blocks supported
  local blockCount = 2
  
  local nSamples, data, fvec, viewRefs
  local sampPeriod, sampSize, parmKind
  local samples = {}
  
  -- parse input lines
  local line = parseCSVLine(file, ';')
  
  -- prepare inputs
  for i = 1, blockCount, 1 do
    
    local svec

    -- read data
    nSamples, sampPeriod, sampSize, parmKind, svec = readInputs(line[2+(i-1)*3] .. settings.parExt, line[3+(i-1)*3] + 1, line[4+(i-1)*3] + 1)
    
    -- concat data
    table.insert(samples, line[4+(i-1)*3] - line[3+(i-1)*3] + 1)
    if i == 1 then
      fvec = svec:clone()
    else
      fvec = torch.cat(fvec, svec:clone(), 1)
    end

  end  
  
  -- prepare references
  if prepareRefs then
    local typeD = tonumber(line[1])
    viewRefs = torch.Tensor(fvec:size(1)):zero()
  
    -- predefined outputs for each type - (0 - SN; 1 - NS; 2 - SS; 3 - NN)
    local classA, classB, classC, classD
    if typeD == 0 then
      classA = 0; classB = 4; classC = 3; classD = 1
    elseif typeD == 1 then
      classA = 1; classB = 5; classC = 2; classD = 0
    elseif typeD == 2 then
      classA = 0; classB = 0; classC = 0; classD = 0
    elseif typeD == 3 then
      classA = 1; classB = 1; classC = 1; classD = 1
    end 

    -- fill the output vector
    viewRefs[{{1, samples[1] - settings.seqL}}] = classA
    viewRefs[{{samples[1] - settings.seqL + 1, samples[1]}}] = classB
    viewRefs[{{samples[1] + 1, samples[1] + settings.seqR}}] = classC
    viewRefs[{{samples[1] + settings.seqR + 1, fvec:size(1)}}] = classD
  end
  
  return fvec:size(1), sampPeriod, sampSize, parmKind, fvec, viewRefs
  
end



-- function reading DNN inputs & outputs - view (SCH version - v2)
function readViewV2(file, prepareRefs)
  
  local nSamples, viewRefs
  local samples = {}
  
  -- parse input lines
  local line = parseCSVLine(file, ';')
  
  local sampPeriod, sampSize, parmKind, fvec
  
  nSamples, sampPeriod, sampSize, parmKind, fvec = readInputs(line[1] .. settings.parExt, line[2] + 1, line[4] + 1)

  if prepareRefs then
    local outClass = 1
    viewRefs = torch.Tensor(fvec:size(1)):zero()
    
    if line[7] and line[8] then
      if line[7] == 'm' and line[8] == 'm' then
        outClass = 2
      elseif line[7] == 'f' and line[8] == 'f' then
        outClass = 3
      else
        outClass = 1
      end
    end
    
    viewRefs[{{line[3] - line[2] - settings.seqL + 1, line[3] - line[2] + settings.seqR}}] = outClass
    
    -- i = 0
    -- viewRefs[{{line[3] - line[2] - 50 + 1, line[3] - line[2] + 50}}]:apply(function(x)
    --   i = i + 1
    --   return i
    -- end)  
  end

  return fvec:size(1), sampPeriod, sampSize, parmKind, fvec, viewRefs

end



-- function reading DNN references
function readRefs(file, nSamples)
  
  -- switch to correct reading
  if settings.refType == "akulab" then
    return readAkulab(file, nSamples):float()
  elseif settings.refType == "rec-mapped" then
    return readRecMapped(file, nSamples)
  else
    error('RefType: not supported')
  end   
  
end



-- function reading DNN references - akulab
function readAkulab(file, nSamples)
  
  -- check if the file exists
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!')
  end	
  
  local f = torch.DiskFile(file, 'r')
  
  f:binary()
  local references = f:readInt(nSamples) 
  f:close()
  
  return torch.IntTensor(references)
  
end



-- function reading DNN references - rec-mapped
function readRecMapped(file, nSamples)

  -- check if the file exists
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!')
  end
  
  local refs = {}
  for line in io.lines(file) do
    local splitters = split(line, " ")

    for i = splitters[1], splitters[2]-1, 1 do
      refs[i+1] = splitters[3]
    end
  end

  local refTensor = torch.Tensor(refs)
  
  --if settings.dnnAlign == 1 then 
    refTensor = torch.Tensor(refTensor[{{1, nSamples}}])
  --end
  
  return refTensor
  
end



-- function loading stats (mean / std)
function readStat(file)
  
  -- check if file exists
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!')
  end
  
  local f = torch.DiskFile(file, 'r')
  local stat = f:readFloat(settings.inputSize)
  stat = torch.Tensor(stat, 1, torch.LongStorage{settings.inputSize})
  
  return stat
  
end



-- function reading mean and std
function readStats()
  
  local mean = readStat(settings.outputFolder .. settings.statsFolder .. '/mean.list')
  local std = readStat(settings.outputFolder .. settings.statsFolder .. '/std.list')
  
  return mean, std
  
end



-- function reading framestats
function readFramestats(file)
  
  -- check if file exists  
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!')
  end
  
  local framestats = {}
  local count
  
  local counter = 1
  for line in io.lines(file) do
    local splitters = split(line, " ")
    framestats[counter] = splitters[2]
    counter = counter + 1
    if not splitters[2] then
      count = line
    end
  end

  framestats = torch.FloatTensor(framestats)
  
  return framestats, count
  
end



-- function loading package dataset
function readPackage(listName)

  local path = settings.outputFolder .. settings.packageFolder
  
  -- read inputs
  local nSamples, sampPeriod, sampSize, parmKind, fvec = readInputs(path .. "inp-" .. listName)
  local refs = readAkulab(path .. "ref-" .. listName, nSamples):float()
  local info = readAkulab(path .. "info-" .. listName, 3)
  local nSamplesList = torch.totable(readAkulab(path .. "samp-" .. listName, info[3]))
  
  return nSamples, sampPeriod, sampSize, parmKind, fvec, refs, info[2], nSamplesList
  
end



-- function saving filelist
function saveFilelist(file, list)
  
  io.output(file)
  
  -- fake first line of views for SAD/SCH
  if settings.inputView > 0 then
    io.write(file, "\n")
  end
  
  for i = 1, #list, 1 do
    io.write(list[i], "\n")
  end
  
  io.flush()
  io.close()
  
end



-- function saving package filelists
function savePackageFilelists(filelist)
  
  local trainLists = {}
  local trainList = readFilelist(filelist)
  for i = 1, settings.packageCount, 1 do
    trainLists[i] = {}
  end
  for i = 1, #trainList, 1 do
    table.insert(trainLists[(i%settings.packageCount)+1], trainList[i])
  end
  for i = 1, settings.packageCount, 1 do
    saveFilelist(settings.outputFolder .. settings.packageFolder .. "pckg" .. i .. ".list", trainLists[i])
  end
  
end



-- function saving stats (mean / std)
function saveStat(file, stat)
  
  local f = torch.DiskFile(file, 'w')
  
  for v = 1, stat:size(1), 1 do
    f:writeFloat(stat[v])
  end
  
  f:close()
  
end



-- function saving mean and std
function saveStats(mean, std)
  
  saveStat(settings.outputFolder .. settings.statsFolder .. '/mean.list', mean)
  saveStat(settings.outputFolder .. settings.statsFolder .. '/std.list', std)
  
end



-- function saving framestat
function saveFramestat(file, stat, version)

  io.output(file)
  
  if version == "ntx3" then
    local count = 0
    for v = 1, #stat, 1 do
      io.write(v-1, " ", stat[v], "\n")
      count = count + stat[v]
    end
    io.write(count, "\n")
    
  elseif version == "ntx4" then   -- log (count / counts)
    local totalCount = 0
    for v = 1, #stat, 1 do
      totalCount = totalCount + stat[v]
    end
    
    local value
    for v = 1, #stat, 1 do
      value = math.log(stat[v] / totalCount)
      io.write(value, "\n")
    end
  else
    error('FramestatsVersion: not supported')
  end
    
  io.flush()
  io.close()

end



-- function saving framestats (ntx43 & ntx4 version)
function saveFramestats(framestats, append)
  
  if append then
    saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestats-' .. append .. '.list', framestats, 'ntx3')
    saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestatsV4-' .. append .. '.list', framestats, 'ntx4')
  else
    saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestats.list', framestats, 'ntx3')
    saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestatsV4.list', framestats, 'ntx4')
  end
  
end



-- function saving HTK header
function saveHTKHeader(file, setSize, lkl)
  
  file:writeInt(setSize)
  file:writeInt(100000)
  if lkl then
    file:writeShort(settings.outputSize * 4)
  else
    file:writeShort(settings.inputSize * 4)
  end
  file:writeShort(9)
  
end



-- function saving nnet file
function saveModel(ifile, ofile)
  
  local mlp = torch.load(ifile)
  local linearNodes = mlp:findModules('nn.Linear')
  local f = torch.DiskFile(ofile, "w")
  f:binary()
  f:writeInt(0x54454E4e)
  f:writeInt(0)
  local isize = linearNodes[1].weight:size(2)
  f:writeInt(isize)
  local noLayers = #linearNodes
  f:writeInt(noLayers)
  for i = 1, noLayers do
    local noNeurons = linearNodes[i].weight:size(1)
    f:writeInt(noNeurons)
  end
  for i = 1, noLayers do
    local stor = linearNodes[i].bias:float():storage()
    f:writeFloat(stor)
  end      
  for i = 1, noLayers do
    local stor = linearNodes[i].weight:float():storage()
    f:writeFloat(stor)
  end
  f:close()
  
end



-- function saving package data
function savePackage(cache, nSamples, totalSamples, filesCount, listName)
  
  local path = settings.outputFolder .. settings.packageFolder
  os.execute("mkdir -p " .. path)
  
  local globalSamples = totalSamples + filesCount * (settings.seqL + settings.seqR)
  
  saveCacheHTK(path .. "inp-" .. listName, cache, globalSamples)
  saveCacheAkulab(path .. "ref-" .. listName, cache)
  saveIntTable(path .. "samp-" .. listName, nSamples)
  saveIntTable(path .. "info-" .. listName, {globalSamples, totalSamples, filesCount})
  
end



-- function saving input data (table of xd tensors) to htk format file
function saveCacheHTK(file, cache, globalSamples)
  
  local f = torch.DiskFile(file, "w")
  f:binary()
  
  saveHTKHeader(f, globalSamples)
  
  for i = 1, #cache, 1 do
    f:writeFloat(cache[i].inp:storage())
  end
  
  f:close()
  
end  



-- function saving refs (table of 1d tensor) to akulab format file
function saveCacheAkulab(file, cache)
  
  local f = torch.DiskFile(file, "w")
  f:binary()
      
  for i = 1, #cache, 1 do
    f:writeInt(cache[i].out:int():storage())
  end

  f:close()
  
end



-- function saving 1d table to file as ints
function saveIntTable(file, table)
  
  local f = torch.DiskFile(file, "w")
  f:binary()
      
  for i = 1, #table, 1 do
    f:writeInt(table[i])
  end
  
  f:close()
  
end


