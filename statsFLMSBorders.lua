
require 'torch'
require 'math'
require 'xlua'
require 'logroll'

function Stats(fname, settings)
  
  local stats = {
    fname = fname;
    settings = settings;  
  }
  
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/stats.log');
  plog = logroll.print_logger();
  log = logroll.combine(flog, plog);
  
  if not paths.filep(fname) then  
    log.error('File ' .. fname .. ' does not exist!');
  end

  torch.setdefaulttensortype('torch.FloatTensor');
   
  log.info('Computing mean and variance for ' .. fname);
  
  local begin = sys.clock();
  
  stats.fileList = {};
  stats.nSamples = 0;
  stats.mean = torch.Tensor(settings.inputSize):zero():double();
  stats.var = torch.Tensor(settings.inputSize):zero():double();
  stats.CMSmean = torch.Tensor(settings.inputSize):zero():double();
  stats.CMSvar = torch.Tensor(settings.inputSize):zero():double();
  
  for line in io.lines(fname) do
    line = line:gsub(settings.inputPath, settings.mfccPath);
    table.insert(stats.fileList, line);
  end
  
  for file = 1, #stats.fileList, 1 do    
    local line = stats.fileList[file] ;
    local f = torch.DiskFile(line .. settings.mfccExt, 'r');
    f:binary();
    local nSamples = f:readInt();
    local sampPeriod = f:readInt();
    local sampSize = f:readShort();
    local parmKind = f:readShort();
    local s = f:readFloat(nSamples * sampSize / 4);
        
    f:close();
    
    local fvec = torch.Tensor(s, 1, torch.LongStorage{nSamples, sampSize / 4});
    
    if (settings.borders == 1) then
      local pre = torch.Tensor(s, 1, torch.LongStorage{1, sampSize / 4});
      local post = torch.Tensor(s, (fvec:size(1) - 1) * (sampSize / 4) + 1, torch.LongStorage{1, sampSize / 4});   
      pre = pre:repeatTensor(settings.seqL, 1);
      post = post:repeatTensor(settings.seqR, 1);   
      fvec = torch.cat(pre, torch.cat(fvec, post, 1), 1);
      nSamples = nSamples + settings.seqL + settings.seqR;
    end
    
    local cms = torch.Tensor(nSamples, sampSize / 4):zero();
    local cmsCounter;
    
    local startIndex, endIndex;
    
    for i = 1, nSamples, 1 do
      
      cmsCounter = settings.cms + 1;
      
      startIndex = i - (settings.cms/2);
      endIndex = i + (settings.cms/2);
      
      if(startIndex < 1) then
        cmsCounter = cmsCounter + startIndex - 1;
        startIndex = 1;
      end
      
      if(endIndex > nSamples) then
        cmsCounter = cmsCounter - (endIndex - nSamples);
        endIndex = nSamples;
      end
         
      cms[{ i, {} }] = (torch.sum(fvec[{ {startIndex, endIndex}, {} }], 1)):div(cmsCounter);
      
    end
    
    fvec:add(-cms);
    
    stats.mean:add(torch.sum(fvec, 1):double());
    stats.var:add(torch.sum(torch.pow(fvec, 2):double(), 1));

    stats.nSamples = stats.nSamples + fvec:size(1);
  end   

  stats.mean:div(stats.nSamples);
  stats.var:div(stats.nSamples);
  stats.var:add(-torch.pow(stats.mean, 2));
  stats.var:sqrt();
  stats.mean = stats.mean:float();
  stats.var = stats.var:float();
  
  log.info('Mean and variance completed in ' .. sys.clock() - begin);

  function stats:exportStats();
    
    local output = settings.outputFolder .. settings.statsFolder;
    
    local f = torch.DiskFile(output .. '/mean-' .. stats.fname, 'w');
    for v = 1, stats.mean:size(1), 1 do
      f:writeFloat(stats.mean[v]);
    end
    f:close()
    
    f = torch.DiskFile(output .. '/std-' .. stats.fname, 'w');
    for v = 1, stats.mean:size(1), 1 do
      f:writeFloat(stats.var[v]);
    end
    f:close();
    
    log.info('Stats exported in ' .. output .. '/');
    
  end

  return stats;
  
end

