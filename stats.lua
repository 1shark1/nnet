
-- LM --- Stats

function Stats(fname)
  
  -- initialization
  local stats = {}
  
  -- logs
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/stats.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);
  
  -- check if filelist exists
  if not paths.filep(settings.listFolder .. fname) then  
    flog.error('File ' .. fname .. ' does not exist!');
    error('File ' .. fname .. ' does not exist!');
  end

  -- set tensors to float to calculate stats
  torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  log.info('Computing mean and variance for ' .. fname);
  local begin = sys.clock();
  
  -- stats initialization
  stats.mean = torch.Tensor(settings.inputSize):zero():double();
  stats.var = torch.Tensor(settings.inputSize):zero():double();
  stats.nSamples = 0;
  
  -- load filelist
  local fileList = readFileList(fname);
  
  -- process files one by one
  for file = 1, #fileList, 1 do    
    
    local nSamples, sampPeriod, sampSize, parmKind, data, fvec
    
    -- read input files
    if (settings.inputType == "htk") then
      nSamples, sampPeriod, sampSize, parmKind, data, fvec = readHTK(fileList[file]);
    else
      flog.error('Not implemented');
      error('Not implemented');
    end
    
    -- clone borders
    if (settings.cloneBorders == 1) then
      fvec = cloneBordersInputs(data, fvec);
      nSamples = nSamples + settings.seqL + settings.seqR;
    end
  
    -- compute CMS
    if (settings.applyCMS == 1) then
      local cms = applyCMS(fvec, nSamples);
    end

    -- compute global stats
    stats.mean:add(torch.sum(fvec, 1):double());
    stats.var:add(torch.sum(torch.pow(fvec, 2):double(), 1));
    
    -- compute global number of frames
    stats.nSamples = stats.nSamples + fvec:size(1);

  end   

  -- compute global stats
  stats.mean:div(stats.nSamples);
  stats.var:div(stats.nSamples);
  stats.var:add(-torch.pow(stats.mean, 2));
  stats.var:sqrt();
  stats.mean = stats.mean:float();
  stats.var = stats.var:float();
  
  -- log time needed for computation
  log.info('Mean and variance completed in ' .. sys.clock() - begin);

  -- function exporting stats to file
  function stats:exportStats();

    local output = settings.outputFolder .. settings.statsFolder;
    saveStat(output .. '/mean.list', stats.mean);
    saveStat(output .. '/std.list', stats.var);
    log.info('Stats exported in ' .. output .. '/');
    
  end

  return stats;
  
end
