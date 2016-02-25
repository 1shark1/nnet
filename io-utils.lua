
-- LM --- Input/Output Utils

-- function loading HTK param file
function readHTK(file) 
  
  if not paths.filep(file .. settings.parExt) then  
    error('File ' .. file .. settings.parExt .. ' does not exist!');
  end
  
  local f = torch.DiskFile(file .. settings.parExt, 'r');
  f:binary();
  
  -- read HTK header
  local nSamples = f:readInt();
  local sampPeriod = f:readInt();
  local sampSize = f:readShort();
  local parmKind = f:readShort();
  
  -- read features
  local data = f:readFloat(nSamples * sampSize / 4);    
  f:close();
   
  -- create tensor from input data
  local fvec = torch.Tensor(data, 1, torch.LongStorage{nSamples, sampSize / 4});
  
  return nSamples, sampPeriod, sampSize, parmKind, data, fvec;
  
end

-- function loading view (2 classes)
function readView(file, prepareOutputs)
  
  -- currently supported only 2 blocks (more blocks -> different structure of view file)
  local blockCount = 2;
  
  local nSamples, data, fvec, viewOut;
  local samples = {};
  
  -- parse view
  local line = parseCSVLine(file, ';'); 
  
  -- read blocks and concat data
  for i = 1, blockCount, 1 do
       
    local sampPeriod, sampSize, parmKind, svec;
    
    -- prepare inputs
    if not paths.filep(line[2+(i-1)*3] .. settings.parExt) then  
      error('File ' .. line[2+(i-1)*3] .. settings.parExt .. ' does not exist!');
    end
    
    if (settings.inputType == "htk") then
      nSamples, sampPeriod, sampSize, parmKind, data, svec = readHTK(line[2+(i-1)*3]);
    else
      error('InputType: not supported');
    end
    
    svec = svec[{{line[3+(i-1)*3] + 1, line[4+(i-1)*3] + 1}}];
    table.insert(samples, line[4+(i-1)*3] - line[3+(i-1)*3] + 1);
   
    if(i == 1) then
      fvec = svec:clone();
    else
      fvec = torch.cat(fvec, svec:clone(), 1);
    end
    
  end  
    
  -- prepare outputs
  if (prepareOutputs == 1) then

    local typeD = tonumber(line[1]);
      
    viewOut = torch.Tensor(fvec:size(1)):zero(); 
        
    -- predefined outputs for each type
    -- (0 - SN; 1 - NS; 2 - SS; 3 - NN)
    local classA, classB, classC, classD;
    if(typeD == 0) then
      classA = 0;
      classB = 4;
      classC = 3;
      classD = 1;
    elseif(typeD == 1) then
      classA = 1;
      classB = 5;
      classC = 2;
      classD = 0;
    elseif(typeD == 2) then
      classA = 0;
      classB = 0;
      classC = 0;
      classD = 0;
    elseif(typeD == 3) then
      classA = 1;
      classB = 1;
      classC = 1;
      classD = 1;
    end 

    for i = 1, samples[1] - settings.seqL , 1 do
      viewOut[i] = classA;
    end
    for i = samples[1] - settings.seqL + 1, samples[1] , 1 do
      viewOut[i] = classB;
    end
    for i = samples[1] + 1, samples[1] + settings.seqR, 1 do
      viewOut[i] = classC;
    end
    for i =  samples[1] + settings.seqR + 1, fvec:size(1), 1 do
      viewOut[i] = classD;
    end 
    
  end

  data = torch.view(fvec, settings.inputSize * fvec:size(1)):storage();
  
  return fvec:size(1), sampPeriod, sampSize, parmKind, data, fvec, viewOut;
  
end

-- function saving HTK header
function saveHTKHeader(file, setSize)
  
  file:writeInt(setSize);
  file:writeInt(100000);
  file:writeShort(settings.outputSize * 4);
  file:writeShort(9);
  
end

-- function to save stats
function saveStat(file, stat)
  
  local f = torch.DiskFile(file, 'w');
  
  for v = 1, stat:size(1), 1 do
    f:writeFloat(stat[v]);
  end
  
  f:close()
  
end

-- function loading stats
function readStat(file)
  
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!');
  end

  torch.setdefaulttensortype('torch.FloatTensor');
  
  local f = torch.DiskFile(file, 'r');
  local stat = f:readFloat(settings.inputSize);
  stat = torch.Tensor(stat, 1, torch.LongStorage{settings.inputSize});
  
  torch.setdefaulttensortype('torch.DoubleTensor');
  
  return stat;
  
end

-- function loading filelist
function readFileList(fileList)

  if not paths.filep(fileList) then  
    error('File ' .. fileList .. ' does not exist!');
  end
  
  local file = io.open(fileList, "r");
  io.input(file);
  
  -- if view, skip the info line
  if (settings.inputView == 1) then
    local info = io.read();
  end  
  
  -- load lines to process
  local files = {};
  for line in io.lines() do
    line = string.gsub(line, "%s+", "");
    table.insert(files, line);
  end
  
  io.close(file);
  
  return files;
  
end


-- function reading akulabs
function readAkulab(file, nSamples)
  
  if not paths.filep(file .. settings.refExt) then  
    error('File ' .. file .. settings.refExt .. ' does not exist!');
  end	
  
  local f = torch.DiskFile(file .. settings.refExt, 'r');
  
  f:binary();
  local currentOutput = f:readInt(nSamples);  
  f:close();
  
  return currentOutput;
  
end

-- function reading rec-mapped
function readRecMapped(file, nSamples)
  
  if not paths.filep(file .. settings.refExt) then  
    error('File ' .. file .. settings.refExt .. ' does not exist!');
  end
  
  local refs = {};
  for line in io.lines(file .. settings.refExt) do
    local splitters = split(line, " ");

    for i = splitters[1], splitters[2]-1, 1 do
      refs[i+1] = splitters[3];
    end
  end

  local refTensor = torch.Tensor(refs);
  if (settings.dnnAlign == 1) then 
    refTensor = refTensor[{{1, nSamples}}];
  end
  
  return refTensor;
  
end

-- function saving framestats
function saveFramestats(file, stat)

  io.output(file);

  local count = 0;
  for v = 1, #stat, 1 do
    io.write(v-1, " ", stat[v], "\n");
    count = count + stat[v];
  end
  io.write(count, "\n");
  io.flush();
  io.close();

end

--function loading framestats
function loadFramestats(file)
  
  if not paths.filep(file) then  
    error('File ' .. file .. ' does not exist!');
  end
  
  local framestats = {};
  local count;
  
  local counter = 1;
  for line in io.lines(file) do
    local splitters = split(line, " ");
    framestats[counter] = splitters[2];
    counter = counter + 1;
    if not (splitters[2]) then
      count = line;
    end
  end

  framestats = torch.FloatTensor(framestats);
  
  return framestats, count;
  
end

-- function exporting nnet file
function exportModel(ifile, ofile)
  
  local mlp = torch.load(ifile);
  local linearNodes = mlp:findModules('nn.Linear');
  local f = torch.DiskFile(ofile, "w");
  f:binary();
  f:writeInt(0x54454E4e);
  f:writeInt(0);
  local isize = linearNodes[1].weight:size(2);
  f:writeInt(isize);
  local noLayers = #linearNodes;
  f:writeInt(noLayers); 
  for i = 1, noLayers do
    -- m, n = linearNodes[i].weight:size();
    local noNeurons = linearNodes[i].weight:size(1);
    f:writeInt(noNeurons);
  end
  for i = 1, noLayers do
    local stor = linearNodes[i].bias:float():storage();
    f:writeFloat(stor);
  end      
  for i = 1, noLayers do
    local stor = linearNodes[i].weight:float():storage();
    f:writeFloat(stor);
  end
  f:close();
  
end
