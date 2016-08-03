
-- LM -- Input/Output Utils -- 3/8/16 --



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
  if not paths.filep(file .. settings.parExt) then  
    error('File ' .. file .. settings.parExt .. ' does not exist!')
  end
  
  -- read file
  local f = torch.DiskFile(file .. settings.parExt, 'r')
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

    -- check if the files exist
    if not paths.filep(line[2+(i-1)*3] .. settings.parExt) then  
      error('File ' .. line[2+(i-1)*3] .. settings.parExt .. ' does not exist!')
    end

    -- read data
    nSamples, sampPeriod, sampSize, parmKind, svec = readInputs(line[2+(i-1)*3], line[3+(i-1)*3] + 1, line[4+(i-1)*3] + 1)
    
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

  -- check if the files exist
  if not paths.filep(line[1] .. settings.parExt) then  
    error('File ' .. line[1] .. settings.parExt .. ' does not exist!')
  end
  
  nSamples, sampPeriod, sampSize, parmKind, fvec = readInputs(line[1], line[2] + 1, line[4] + 1)

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
    return readAkulab(file, nSamples)
  elseif settings.refType == "rec-mapped" then
    return readRecMapped(file, nSamples)
  else
    error('RefType: not supported')
  end   
  
end



-- function reading DNN references - akulab
function readAkulab(file, nSamples)
  
  -- check if the file exists
  if not paths.filep(file .. settings.refExt) then  
    error('File ' .. file .. settings.refExt .. ' does not exist!')
  end	
  
  local f = torch.DiskFile(file .. settings.refExt, 'r')
  
  f:binary()
  local references = f:readInt(nSamples) 
  f:close()
  
  return references
  
end



-- function reading DNN references - rec-mapped
function readRecMapped(file, nSamples)

  -- check if the file exists
  if not paths.filep(file .. settings.refExt) then  
    error('File ' .. file .. settings.refExt .. ' does not exist!')
  end
  
  local refs = {}
  for line in io.lines(file .. settings.refExt) do
    local splitters = split(line, " ")

    for i = splitters[1], splitters[2]-1, 1 do
      refs[i+1] = splitters[3]
    end
  end

  local refTensor = torch.Tensor(refs)
  if settings.dnnAlign == 1 then 
    refTensor = refTensor[{{1, nSamples}}]
  end
  
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
    saveFilelist(settings.outputFolder .. settings.logFolder .. "pckg" .. i .. ".list", trainLists[i])
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


