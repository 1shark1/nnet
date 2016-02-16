
-- LM --- Datasets

function Dataset(fname, isFileList, computeFramestats)
  
  -- initialization
  local dataset = {}
  
  -- logs
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/' .. fname .. '.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);
  
  -- check if filelist exists
  if not paths.filep(settings.listFolder .. fname) then  
    flog.error('File ' .. fname .. ' does not exist!');
    error('File ' .. fname .. ' does not exist!');
  end
  
  -- set tensors to float to handle data
  torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  log.info('Preparing dataset: ' .. fname);
  local begin = sys.clock();
  
  -- dataset initialization
  dataset.index = {};
  dataset.nSamplesList = {};
  dataset.cache = {};
  dataset.nSamples = 0;
  
  -- initialize sample count
  local totalSamples = 0;
  
  -- load filelist
  local fileList = {};
  if(isFileList == 1) then
    fileList = readFileList(settings.listFolder .. fname);
  else
    table.insert(fileList, fname);
  end
  
  -- initialize framestats
  local framestats = {};
  if(computeFramestats == 1) then 
    for i = 1, settings.outputSize, 1 do
      framestats[i] = 0;
    end
  end
    
  -- read data from files
  for file = 1, #fileList, 1 do  

    local nSamples, sampPeriod, sampSize, parmKind, data, fvec;
    
    -- read input files
    if (settings.inputType == "htk") then
      nSamples, sampPeriod, sampSize, parmKind, data, fvec = readHTK(fileList[file]);
    else
      flog.error('InputType: not implemented');
      error('InputType: not implemented');
    end
    
    -- fix for DNN alignment by ntx4
    if (settings.dnnAlign == 1) then
      nSamples = nSamples - 1;
    end

    -- read ref outputs
    local currentOutput;
    if (settings.sameFolder == 0) then
      fileList[file] = fileList[file]:gsub(settings.parPath, settings.refPath);
    end
    
    if (settings.refType == "akulab") then
      currentOutput = readAkulab(fileList[file], nSamples);
    elseif (settings.refType == "rec-mapped") then
      currentOutput = readRecMapped(fileList[file], nSamples);
    else
      flog.error('RefType: not implemented');
      error('RefType: not implemented');
    end
    
    -- compute framestats
    if (computeFramestats == 1) then
      for i = 1, currentOutput:size(1), 1 do
        framestats[currentOutput[i]+1] = framestats[currentOutput[i]+1] + 1;
      end
    end

    -- sanity check - input/ref sample size + ntx4 fix
    if (currentOutput:size(1) == nSamples +1) then
      currentOutput = currentOutput[{{1, nSamples}}];
    elseif (currentOutput:size(1) ~= nSamples) then
      flog.error('Nonmatching sample count: ' .. fileList[file]);
      error('Nonmatching sample count' .. fileList[file]);
    end
    
    -- clone borders
    if (settings.cloneBorders == 1) then
      fvec = cloneBordersInputs(data, fvec);
      currentOutput = cloneBordersRefs(currentOutput);
      nSamples = nSamples + settings.seqL + settings.seqR;
    end

    -- compute CMS
    if (settings.applyCMS == 1) then
      local cms = applyCMS(fvec, nSamples);
    end

    -- save CMS processed data to cache table
    fvec = fvec:view(fvec:size(1) * settings.inputSize);
    table.insert(dataset.cache, {inp = fvec, out = currentOutput});
    
    -- save counts of samples per file to table
    nSamples = nSamples - settings.seqL - settings.seqR;
    table.insert(dataset.nSamplesList, nSamples);
    
    -- calculate total samples
    if (nSamples >= 1) then
      totalSamples = totalSamples + nSamples;
    end
    
  end
  
  -- save framestats
  if (computeFramestats == 1) then
    local output = settings.outputFolder .. settings.statsFolder;
    saveFramestats(output .. '/framestats.list', framestats);
  end
  
  -- prepare tensors for data for training
  dataset.index.file = torch.Tensor(totalSamples):int();    -- tensor -> frames vs. file
  dataset.index.pos = torch.Tensor(totalSamples):int();     -- tensor -> frames vs. position in file
  
  -- fill tensors accordingly to allow training
  for ll = 1, #dataset.nSamplesList, 1 do   
    local nSamples = dataset.nSamplesList[ll];
    if (nSamples >= 1) then     -- sanity check
      i = settings.seqL;
      dataset.index.file:narrow(1, dataset.nSamples + 1, nSamples):fill(ll);              -- fe [1 1 1 2 2 2 2 2]
      dataset.index.pos:narrow(1, dataset.nSamples + 1, nSamples):apply(function(x)       -- fe [1 2 3 1 2 3 4 5] + seqL
        i = i + 1;
        return i;
      end);
      dataset.nSamples = dataset.nSamples + nSamples;   -- compute final number of samples
    end
  end  
  
  -- prepare mean & std tensors
  local mean = settings.mean:repeatTensor(settings.seqL + settings.seqR + 1);
  local var = settings.var:repeatTensor(settings.seqL + settings.seqR + 1);
  
  -- log time
  log.info('Dataset prepared in ' .. sys.clock() - begin);
  
  -- return number of samples
  function dataset:size() 
    return dataset.nSamples;
  end
  
  -- return frame + surroundings and akulab
  function dataset:get(i)
    -- identify file
    local fileid = dataset.index.file[i];
    
    -- load file data
    local currentInput = self.cache[fileid].inp;
    local currentOutput = self.cache[fileid].out;
    
    -- find the indices of asked data
    local startIndex = (dataset.index.pos[i] - settings.seqL - 1) * settings.inputSize + 1;
    local endIndex = startIndex + (settings.seqR + 1 + settings.seqL) * settings.inputSize - 1;
    
    -- clone the asked data   
    local inp = currentInput[{{startIndex, endIndex}}]:clone();

    -- normalize
    inp:add(-mean);
    inp:cdiv(var);  
    
    -- prepare the output
    local out = (currentOutput[dataset.index.pos[i]] + 1);
    
    -- return the asked data
    return {inp = inp, out = out};
    
  end
  
  return dataset;
  
end
