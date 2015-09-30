
require 'torch'
require 'logroll'

-- LM

-- Preparation of various datasets for training

function Dataset(fname, settings)
  
  -- initialization
  local dataset = {
    fname = fname;
    settings = settings;
  }
  
  -- logs
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/' .. fname .. '.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);
  
  -- input file does not exist
  if not paths.filep(fname) then  
    log.error('File ' .. fname .. ' does not exist!');
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
  dataset.nFiles = 0;
  dataset.nSamples = 0;
  dataset.currentfile = 0;
  
  -- initialize sample count
  local totalSamples = 0;
  -- read data from files
  for line in io.lines(fname) do
    -- read HTK parametrization
    line = line:gsub(settings.inputPath, settings.mfccPath);
    local f = torch.DiskFile(line .. settings.mfccExt, "r");
    f:binary();
    local nSamples = f:readInt();
    local sampPeriod = f:readInt();
    local sampSize = f:readShort();
    local parmKind = f:readShort();
    local currentInput = f:readFloat(nSamples * sampSize / 4);
    f:close();
    
    -- read akulabels
    line = line:gsub(settings.mfccPath, settings.akuPath);
    f = torch.DiskFile(line .. settings.akuExt, "r");
    f:binary();
    local currentOutput = f:readInt(nSamples);
    f:close();
    
    -- create tensor from input data
    local fvec = torch.Tensor(currentInput, 1, torch.LongStorage{nSamples, sampSize / 4});     
    
    -- clone borders if selected -> inputs & outputs
    if (settings.borders == 1) then
      -- inputs
      local pre = torch.Tensor(currentInput, 1, torch.LongStorage{1, sampSize / 4});
      local post = torch.Tensor(currentInput, (fvec:size(1) - 1) * (sampSize / 4) + 1, torch.LongStorage{1, sampSize / 4});   
      pre = pre:repeatTensor(settings.seqL, 1);
      post = post:repeatTensor(settings.seqR, 1);   
      fvec = torch.cat(pre, torch.cat(fvec, post, 1), 1);
      -- outputs
      local curOut = torch.Tensor(currentOutput:size(1) + settings.seqL + settings.seqR);   
      for i = 1, settings.seqL, 1 do
        curOut[i] = currentOutput[1];
      end  
      for i = settings.seqL + 1, curOut:size(1) - settings.seqR - settings.seqL, 1 do
        curOut[i] = currentOutput[i];
      end  
      for i = curOut:size(1) - settings.seqR - settings.seqL + 1, curOut:size(1) - settings.seqL, 1 do
        curOut[i] = currentOutput[currentOutput:size(1)];
      end    
      currentOutput = curOut;    
      -- get samples
      nSamples = nSamples + settings.seqL + settings.seqR;
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
  
  -- prepare tensors for data for training
  dataset.index.file = torch.Tensor(totalSamples):int();    -- tensor -> frames vs. file
  dataset.index.pos = torch.Tensor(totalSamples):int();     -- tensor -> frames vs. position in file
  
  -- fill tensors accordingly to allow training
  for ll = 1, #dataset.nSamplesList, 1 do   
    local nSamples = dataset.nSamplesList[ll];
    if (nSamples >= 1) then     -- sanity check
      i = settings.seqL;
      dataset.index.file:narrow(1, dataset.nSamples + 1, nSamples):fill(ll);              -- fe [0 0 0 1 1 1 1 1]
      dataset.index.pos:narrow(1, dataset.nSamples + 1, nSamples):apply(function(x)       -- fe [0 1 2 0 1 2 3 4] + seqL
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