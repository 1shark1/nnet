
-- LM -- Input/Output Utils -- 20/5/16 --



-- function reading filelist
function readFilelist(filelist)
  
  -- check if the file exists
  if not paths.filep(filelist) then  
    error('File ' .. filelist .. ' does not exist!')
  end
  
  local file = io.open(filelist, "r")
  io.input(file)
  
  -- if view -> skip the info line
  if settings.inputView == 1 then
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
function readInputs(file)
  
  local nSamples, sampPeriod, sampSize, parmKind, data, fvec
  
  -- switch to correct reading
  if settings.inputType == "htk" then
    nSamples, sampPeriod, sampSize, parmKind, data, fvec = readHTK(file)
  else
    error('InputType: not supported')
  end
  
  return nSamples, sampPeriod, sampSize, parmKind, data, fvec
  
end



-- function reading DNN inputs - HTK format
function readHTK(file) 
  
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
  
  -- ntx4 alignment fix
  if settings.dnnAlign == 1 then
    nSamples = nSamples - 1
  end
  
  -- read features
  local data = f:readFloat(nSamples * sampSize / 4)  
  f:close()

  -- create tensor from input data
  local fvec = torch.Tensor(data, 1, torch.LongStorage{nSamples, sampSize / 4})
  
  return nSamples, sampPeriod, sampSize, parmKind, data, fvec
  
end



-- function reading DNN inputs & outputs - view
function readView(file, prepareRefs)
  
  -- only 2 blocks supported
  local blockCount = 2
  
  local nSamples, data, fvec, viewRefs
  local samples = {}
  
  -- parse input lines
  local line = parseCSVLine(file, ';')
  
  -- prepare inputs
  for i = 1, blockCount, 1 do

    local sampPeriod, sampSize, parmKind, svec

    -- check if the files exist
    if not paths.filep(line[2+(i-1)*3] .. settings.parExt) then  
      error('File ' .. line[2+(i-1)*3] .. settings.parExt .. ' does not exist!')
    end

    -- read data
    nSamples, sampPeriod, sampSize, parmKind, data, svec = readInputs(line[2+(i-1)*3])

    -- concat data
    svec = svec[{{line[3+(i-1)*3] + 1, line[4+(i-1)*3] + 1}}]
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
    for i = 1, samples[1] - settings.seqL, 1 do
      viewRefs[i] = classA
    end
    for i = samples[1] - settings.seqL + 1, samples[1], 1 do
      viewRefs[i] = classB
    end
    for i = samples[1] + 1, samples[1] + settings.seqR, 1 do
      viewRefs[i] = classC
    end
    for i =  samples[1] + settings.seqR + 1, fvec:size(1), 1 do
      viewRefs[i] = classD
    end 

  end

  data = torch.view(fvec, settings.inputSize * fvec:size(1)):storage()
  
  return fvec:size(1), sampPeriod, sampSize, parmKind, data, fvec, viewRefs
  
end



-- function reading DNN references
function readRefs(file, nSamples)
  
  local refs
  
  -- switch to correct reading
  if settings.refType == "akulab" then
    refs = readAkulab(file, nSamples)
  elseif settings.refType == "rec-mapped" then
    refs = readRecMapped(file, nSamples)
  else
    error('RefType: not supported')
  end   
  
  return refs
  
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
function saveFramestats(framestats)
  
  saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestats.list', framestats, 'ntx3')
  saveFramestat(settings.outputFolder .. settings.statsFolder .. '/framestatsV4.list', framestats, 'ntx4')
  
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


