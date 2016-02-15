
-- LM --- Input/Output Utils

-- function loading HTK param file
function readHTK(file) 
  
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

  torch.setdefaulttensortype('torch.FloatTensor');
  
  local f = torch.DiskFile(file, 'r');
  local stat = f:readFloat(settings.inputSize);
  
  torch.setdefaulttensortype('torch.DoubleTensor');
  
  return stat;
  
end

-- function loading filelist
function readFileList(fileList)
  
  local files = {};
  for line in io.lines(fileList) do
    table.insert(files, line);
  end
  
  return files;
  
end


-- function reading akulabs
function readAkulab(file, nSamples)
  
  local f = torch.DiskFile(file .. settings.refExt, 'r');
  
  f:binary();
  local currentOutput = f:readInt(nSamples);  
  f:close();
  
  return currentOutput;
  
end

-- function reading rec-mapped
function readRecMapped(file, nSamples)
  
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
  
  local framestats = {};
  
  local counter = 1;
  for line in io.lines(file) do
    local splitters = split(line, " ");
    framestats[counter] = splitters[2];
    counter = counter + 1;
  end
  
  return framestats;
  
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
