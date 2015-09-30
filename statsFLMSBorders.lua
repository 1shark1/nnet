
require 'torch'
require 'math'
require 'xlua'
require 'logroll'

-- LM

-- Computation of train data stats - mean, std

function Stats(fname, settings)
  
  -- initialization
  local stats = {
    fname = fname;
    settings = settings;  
  }
  
  -- logs
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/stats.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);
  
  -- input file does not exist
  if not paths.filep(fname) then  
    log.error('File ' .. fname .. ' does not exist!');
  end

  -- set tensors to float to calculate stats
  torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  log.info('Computing mean and variance for ' .. fname);
  local begin = sys.clock();
  
  -- stats initialization
  stats.fileList = {};
  stats.nSamples = 0;
  stats.mean = torch.Tensor(settings.inputSize):zero():double();
  stats.var = torch.Tensor(settings.inputSize):zero():double();
  stats.CMSmean = torch.Tensor(settings.inputSize):zero():double();
  stats.CMSvar = torch.Tensor(settings.inputSize):zero():double();
  
  -- read filelist
  for line in io.lines(fname) do
    line = line:gsub(settings.inputPath, settings.mfccPath);
    table.insert(stats.fileList, line);
  end
  
  -- process param files one by one [HTK format]
  for file = 1, #stats.fileList, 1 do    
    local line = stats.fileList[file] ;
    local f = torch.DiskFile(line .. settings.mfccExt, 'r');
    f:binary();
    -- read HTK header
    local nSamples = f:readInt();
    local sampPeriod = f:readInt();
    local sampSize = f:readShort();
    local parmKind = f:readShort();
    -- read features
    local s = f:readFloat(nSamples * sampSize / 4);    
    f:close();
    
    -- create tensor from input data
    local fvec = torch.Tensor(s, 1, torch.LongStorage{nSamples, sampSize / 4});
    
    -- clone borders if selected
    if (settings.borders == 1) then
      local pre = torch.Tensor(s, 1, torch.LongStorage{1, sampSize / 4});
      local post = torch.Tensor(s, (fvec:size(1) - 1) * (sampSize / 4) + 1, torch.LongStorage{1, sampSize / 4});   
      pre = pre:repeatTensor(settings.seqL, 1);
      post = post:repeatTensor(settings.seqR, 1);   
      fvec = torch.cat(pre, torch.cat(fvec, post, 1), 1);
      nSamples = nSamples + settings.seqL + settings.seqR;
    end
    
    -- compute CMS
    local cms = torch.Tensor(nSamples, sampSize / 4):zero();
    local cmsCounter;
    -- indices for computation
    local startIndex, endIndex;
    
    for i = 1, nSamples, 1 do
      -- normal data
      cmsCounter = settings.cms + 1;
      startIndex = i - (settings.cms/2);
      endIndex = i + (settings.cms/2);
      -- start
      if(startIndex < 1) then
        cmsCounter = cmsCounter + startIndex - 1;
        startIndex = 1;
      end
      -- end
      if(endIndex > nSamples) then
        cmsCounter = cmsCounter - (endIndex - nSamples);
        endIndex = nSamples;
      end
      -- compute CMS for given frame   
      cms[{ i, {} }] = (torch.sum(fvec[{ {startIndex, endIndex}, {} }], 1)):div(cmsCounter);
    end
    
    -- apply CMS
    fvec:add(-cms);
    
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


  -- export stats to file
  function stats:exportStats();

    local output = settings.outputFolder .. settings.statsFolder;
    
    -- mean  
    local f = torch.DiskFile(output .. '/mean-' .. stats.fname, 'w');
    for v = 1, stats.mean:size(1), 1 do
      f:writeFloat(stats.mean[v]);
    end
    f:close()
    
    -- std
    f = torch.DiskFile(output .. '/std-' .. stats.fname, 'w');
    for v = 1, stats.mean:size(1), 1 do
      f:writeFloat(stats.var[v]);
    end
    f:close();
    
    log.info('Stats exported in ' .. output .. '/');
    
  end

  return stats;
  
end

