
-- LM -- Stats



function Stats(list)
  
  -- initialization
  local stats = {}
  
  -- logs
  local flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/stats.log')
  local plog = logroll.print_logger()
  local log = logroll.combine(flog, plog)
  
  -- log & timer
  log.info('Computing mean and variance for ' .. list)
  local begin = sys.clock()
  
  -- stats initialization
  stats.mean = torch.Tensor(settings.inputSize):zero():double()
  stats.std = torch.Tensor(settings.inputSize):zero():double()
  stats.nSamples = 0
  
  -- load filelist
  local filelist = readFilelist(list)
  
  -- process files one by one
  for file = 1, #filelist, 1 do    

    flog.info('Processing file: ' .. filelist[file])

    local fvec

    -- read input files
    if settings.inputView > 0 then
      _, _, _, _, fvec = readView(filelist[file])
    else
      _, _, _, _, fvec = readInputs(filelist[file] .. settings.parExt)
    end
  
    -- apply CMS
    if settings.applyCMS == 1 then
      applyCMS(fvec)
    end   

    -- compute global stats
    stats.mean:add(torch.sum(fvec, 1):double())
    stats.std:add(torch.sum(torch.pow(fvec, 2):double(), 1))

    -- compute global number of frames
    stats.nSamples = stats.nSamples + fvec:size(1)

  end   

  -- compute global stats
  stats.mean:div(stats.nSamples)
  stats.std:div(stats.nSamples)
  stats.std:add(-torch.pow(stats.mean, 2))
  stats.std:sqrt()
  local rand = (torch.rand(settings.inputSize) * 1e-15):double()
  stats.mean = (stats.mean+rand):float()
  stats.std = (stats.std+rand):float()

  -- log time needed for computation
  log.info('Mean and variance completed in ' .. sys.clock() - begin)

  return stats
  
end


