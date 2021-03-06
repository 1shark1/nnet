
-- LM -- DNN Training



-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'optim'
require 'logroll'
require 'gnuplot'

-- require settings
local setts = 'settings'
if arg[1] then 
  setts = string.gsub(arg[1], ".lua", "")
end
assert(require(setts))

torch.manualSeed(1)

-- set default tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- initialize settings
settings = Settings() 
  
-- add path to scripts
if settings.scriptFolder then  
  package.path = package.path .. ";" .. settings.scriptFolder .. "?.lua"
end

-- program requires
require 'utils'
require 'io-utils'
require 'set-utils'
require 'nn-utils'
require 'stats'
require 'dataset'

-- compute/load global stats on training set
local stats
if settings.loadStats == 1 then
  settings.mean, settings.std = readStats()
elseif settings.computeStats == 1 then
  stats = Stats(settings.listFolder .. settings.lists[1])
  settings.mean = stats.mean
  settings.std = stats.std
end

-- save stats to files
saveStats(settings.mean, settings.std)

-- prepare train set
local sets = {}
if settings.packageCount <= 1 then        -- 0 or 1 packages - load data to RAM
  if settings.loadPackage == 1 then
    table.insert(sets, Dataset(settings.outputFolder .. settings.packageFolder .. "pckg1.list", true, false, true, false))
  elseif settings.savePackage == 1 then
    if settings.scriptFolder then
      os.execute("th " .. settings.scriptFolder .. "save-pckg.lua " .. setts)
    else
      os.execute("th save-pckg.lua " .. setts)
    end
    table.insert(sets, Dataset(settings.outputFolder .. settings.packageFolder .. "pckg1.list", true, false, true, false))
  else
    table.insert(sets, Dataset(settings.listFolder .. settings.lists[1], true, true))
    settings.packageCount = 1
    saveFramestats(sets[1].framestats)
  end  
elseif settings.packageCount > 1 then     -- more packages -> load data during training
  os.execute("mkdir -p " .. settings.outputFolder .. settings.packageFolder)
  table.insert(sets, -1)
  if settings.loadPackage == 0 then
    savePackageFilelists(settings.listFolder .. settings.lists[1])
  
    -- save packages
    if settings.savePackage == 1 then
      os.execute("th save-pckg.lua " .. setts)
    end
  end
end

-- prepare other lists
for i = 2, #settings.lists, 1 do
  table.insert(sets, Dataset(settings.listFolder .. settings.lists[i], true, false))
end
  
-- initialize logs
local flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train-' .. settings.lists[1] .. '.log')
local plog = logroll.print_logger()
local log = logroll.combine(flog, plog)
  
-- compute number of batches per sets
local noBatches = {}
if settings.packageCount == 1 then 
  table.insert(noBatches, (sets[1]:size() - sets[1]:size() % settings.batchSize) / settings.batchSize)
elseif settings.packageCount > 1 then
  table.insert(noBatches, -1) 
end
  
-- all batch sizes for evaluation sets
for i = 2, #sets, 1 do
  table.insert(noBatches, math.ceil(sets[i]:size() / settings.batchSize))
end

-- require cuda
if settings.cuda == 1 then 
  require 'cunn'
  require 'cutorch' 
end

-- prepare the network
local model
if settings.startEpoch == 0 then
  model = buildDNN()
else  -- load epoch
  if paths.filep(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod") then  
    model = torch.load(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod")
    log.info('Epoch ' .. settings.startEpoch .. ' loaded')
  else
    error('Epoch ' .. settings.startEpoch .. ".mod" .. ' does not exist!')
  end
end

-- loss function
local criterion = getCriterion()

-- cuda
if settings.cuda == 1 then
  model:cuda()
  criterion:cuda()
  flog.info("Using CUDA")
else
  flog.info("Using CPU")
end

-- set structure for evaluation
local errorTable = {}
for i = 2, #settings.lists, 1 do
  table.insert(errorTable, {})
end

-- retrieve parameters and gradients
local parameters, gradParameters = model:getParameters()

-- optim inits
local config = getOptimParams()

-- process by epochs
for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do
  
  -----------TRAINING-----------

  -- mode set to training
  model:training()
  
  -- timer
  local etime = sys.clock()
  
  -- log
  log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs)
  
  -- shuffle packages
  local setOrder = torch.randperm(settings.packageCount)
  
  -- package training
  for noPackage = 1, settings.packageCount, 1 do
    
    if settings.packageCount > 1 then
      
      -- prepare package
      -- free memory
      sets[1] = {}
      collectgarbage()
      
      -- prepare package dataset    
      if settings.savePackage == 1 or settings.loadPackage == 1 then
        sets[1] = Dataset(settings.outputFolder .. settings.packageFolder .. "pckg" .. setOrder[noPackage] .. ".list", true, false, true, false)
      else
        if epoch == 1 then
          sets[1] = Dataset(settings.outputFolder .. settings.packageFolder .. "pckg" .. setOrder[noPackage] .. ".list", true, true)
        else
          sets[1] = Dataset(settings.outputFolder .. settings.packageFolder .. "pckg" .. setOrder[noPackage] .. ".list", true, false)
        end
      end   
      
      -- compute number of batches for training
      noBatches[1] = (sets[1]:size() - sets[1]:size() % settings.batchSize) / settings.batchSize
      
      -- log
      log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs .. " (package " .. noPackage .. "/" .. settings.packageCount .. ")")
    
    end
  
    -- shuffle data
    local shuffle
    if settings.shuffle == 1 then
      shuffle = torch.randperm(sets[1]:size(), 'torch.LongTensor')
    end
    
    -- mini batch training
    for noBatch = 1, noBatches[1], 1 do

      -- prepare inputs & outputs tensors    
      local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero()
      local targets = torch.Tensor(settings.batchSize):zero()
    
      -- process batches
      for i = 1, settings.batchSize, 1 do
    
        -- pick frame 
        local index = (noBatch - 1) * settings.batchSize + i   
        if settings.shuffle == 1 then
          index = shuffle[index]
        end
    
        -- retrieve data for selected frame, fill input arrays for training       
        local ret = sets[1][index]
        inputs[i] = ret.inp
        targets[i] = ret.out     

      end
    
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
        -- get new parameters
        if x ~= parameters then parameters:copy(x) end
    
        -- zero the accumulation of the gradients
        gradParameters:zero()
    
        -- cuda neccessities
        if settings.cuda == 1 then
          inputs = inputs:cuda()
          targets = targets:cuda()
        end
        
        -- forward propagation     
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
    
        -- back propagation
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
    
        return f, gradParameters
      end  
    
      -- pick optimization
      config = getOptimParams()
      if settings.optimization == "sgd" then
        optim.sgd(feval, parameters, config)
      elseif settings.optimization == "other" then
        optim.sgd(feval, parameters, config) 
      else
        error('Optimization: not supported')
      end

    end

  end
  
  -- logs & export model
  plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs)
  torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", model)
  if settings.exportNNET == 1 then
    saveModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet")
  end
  log.info("Epoch: " .. epoch .. "/".. settings.noEpochs .. " completed in " .. sys.clock() - etime) 
  
  -----------TESTING-----------

  -- mode set to evaluation
  model:evaluate()
  
  -- log
  plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs)
  
  -- process evaluation lists
  for i = 2, #settings.lists, 1 do
    
    -- open file for saving confusion matrix info
    if settings.confusionMatrixInfo == 1 and settings.noEpochs == epoch then
      io.output(settings.outputFolder .. settings.logFolder .. "/conf-" .. settings.lists[i] .. "-" .. epoch .. ".stats")
    end
    
    -- initialize evaluation
    local err_mx = 0
    local all = 0  
    
    -- mini batch processing
    for j = 1, noBatches[i], 1 do  
      -- last batch fix
      local batchSize = settings.batchSize
      if j == noBatches[i] then
        batchSize = sets[i]:size() - ((noBatches[i]-1) * settings.batchSize)
      end

      -- prepare inputs & outputs tensors 
      local inputs = torch.Tensor(batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero()
      local targets = torch.Tensor(batchSize):zero()
  
      -- process batches
      for k = 1, batchSize, 1 do
        -- pick frame and obtain data
        local index = (j - 1) * settings.batchSize + k
        local ret = sets[i][index]
        inputs[k] = ret.inp
        targets[k] = ret.out
      end

      -- cuda neccessities
      if settings.cuda == 1 then
        inputs = inputs:cuda()
      end     

      -- forward pass
      local outputs = model:forward(inputs)
    
      -- cuda neccessities
      if settings.cuda == 1 then
        outputs = outputs:typeAs(targets)
      end         

      -- frame error evaluation
      local _, mx = outputs:max(2)       
      mx = mx:typeAs(targets)      
      err_mx = err_mx + torch.sum(torch.ne(mx, targets))
      all = all + batchSize     
      
      -- save confusion matrix info
      if settings.confusionMatrixInfo == 1 and settings.noEpochs == epoch then
        for k = 1, batchSize, 1 do
          io.write(torch.totable(mx[k])[1], ";", targets[k], "\n")
        end
      end
    end
    
    -- close confusion matrix info file
    if settings.confusionMatrixInfo == 1 and settings.noEpochs == epoch then
      io.flush()
      io.close()
    end
    
    -- save error rate for graph and log
    local err = 100 * err_mx / all
    table.insert(errorTable[i-1], err)
    log.info("Set " .. settings.lists[i] .. " - epoch: " .. epoch .. "/".. settings.noEpochs .. " - err = " .. err)
  end

  -- draw frame error rate graph
  if settings.drawERRs == 1 then
    gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errs-' .. settings.lists[1] .. '.png')
    if #settings.lists - 1 == 1 then 
      gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'})
    elseif #settings.lists - 1 == 2 then 
      gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'})
    elseif #settings.lists - 1 == 3 then 
      gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'}, {settings.lists[4], torch.Tensor(errorTable[3]), '-'})
    else
      flog.error('GNUPlot: not supported')
    end    
    gnuplot.title('Error rates')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('error rate [%]')
    gnuplot.plotflush()
  end
end
