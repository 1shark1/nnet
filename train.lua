
-- LM --- DNN Train

-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'

-- require settings
if (arg[1]) then
  assert(require(arg[1]));
else
  require 'settings'
end

-- initialize settings
settings = Settings();   
  
-- require stats if necessary  
if (settings.computeStats == 1) then
  require 'stats'
end

-- program requires
require 'dataset'
require 'utils'
require 'io-utils'
require 'set-utils'
require 'nn-utils'

-- load modules for cuda if selected
if (settings.cuda == 1) then
  require 'cunn'
  require 'cutorch'  
end

-- compute and export stats on train set
if (settings.computeStats == 1) then
  stats = Stats(settings.lists[1]);  
  stats:exportStats();    
  settings.mean = stats.mean;
  settings.var = stats.var;
end

-- prepare train dataset
sets = {};
if (settings.exportFramestats == 1) then
  table.insert(sets, Dataset(settings.lists[1], 1, settings.exportFramestats));
else
  table.insert(sets, Dataset(settings.lists[1], 1, 0));
end

-- prepare other sets for validation/testing
if (#settings.lists > 0) then
  for i = 2, #settings.lists, 1 do
    table.insert(sets, Dataset(settings.lists[i], 1, 0));
  end
end

-- initialize logs
flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

-- compute number of batches per sets
noBatches = {};
for i = 1, #sets, 1 do
  table.insert(noBatches, (sets[i]:size() - sets[i]:size() % settings.batchSize) / settings.batchSize);
end

-- initialize the network
if (settings.startEpoch == 0) then
  if (settings.model == "classic") then
    model = buildFFModel();
  elseif (settings.model == "residual") then
    model = buildResidualModel();
  else
    flog.error('Not implemented');
    error('Not implemented');
  end
-- load epoch to continue training 
else
  if paths.filep(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod") then  
    model = torch.load(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod");
    log.info('Epoch  ' .. settings.startEpoch .. ' loaded');
  else
    log.error('Epoch ' .. settings.startEpoch .. ".mod" .. ' does not exist!');
  end
end

-- cuda on/off
if (settings.cuda == 1) then 
  model:cuda();
  modelC = nn.Sequential();
  modelC:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));
  modelC:add(model);
  modelC:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'));    
  flog.info("Using CUDA"); 
else
  modelC = model;   
  flog.info("Using CPU");
end

-- set structure for evaluation
if (#settings.lists > 0) then
  errorTable = {};
  for i = 2, #settings.lists, 1 do
    table.insert(errorTable, {});
  end
end

-- DNN

-- criterion 
criterion = nn.ClassNLLCriterion();

for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do
  
  -- TRAINING
  
  -- timer per epoch - start
  local etime = sys.clock();  

  -- shuffle data
  if (settings.shuffle == 1) then
    shuffle = torch.randperm(sets[1]:size(), 'torch.LongTensor');
  end

  -- mode training
  modelC:training()
  
  -- log
  log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  -- training per batches
  for noBatch = 1, noBatches[1], 1 do

    -- prepare inputs & outputs tensors    
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    -- process batches
    for i = 1, settings.batchSize, 1 do
      
      -- pick frame 
      local index = (noBatch - 1) * settings.batchSize + i;     
      if (settings.shuffle == 1) then
        index = shuffle[index];
      end
      
      -- retrieve data for selected frame, fill input arrays for training
      ret = sets[1]:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;

    end
  
    -- forward propagation
    criterion:forward(modelC:forward(inputs), targets);
	
    -- zero the accumulation of the gradients
    modelC:zeroGradParameters();
  
    -- back propagation
    modelC:backward(inputs, criterion:backward(modelC.output, targets));
    
    -- update parameters
    if (settings.lrDecayActive == 1) then 
      learningRate = settings.learningRate / (1 + (epoch - 1) * settings.lrDecay); 
      modelC:updateParameters(learningRate);
    else 
      modelC:updateParameters(settings.learningRate);  
    end
    
  end
  
  -- logs & export model
  plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs);
  torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", modelC);
  if (settings.exportNNET == 1) then
    exportModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet");
  end
  log.info("Epoch: " .. epoch .. "/".. settings.noEpochs .. " completed in " .. sys.clock() - etime);  
  
  -- EVALUATION
  
  -- mode evaluation
  modelC:evaluate();

  plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs);
  if (#settings.lists > 0) then   
    for i = 2, #settings.lists, 1 do
      
      err_mx = 0;
      all = 0;    
      for j = 1, noBatches[i], 1 do
        
        -- prepare inputs & outputs tensors 
        local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
        local targets = torch.Tensor(settings.batchSize):zero();
        
        -- process batches
        for k = 1, settings.batchSize, 1 do
          -- pick frame and obtain data
          local index = (noBatches[i] - 1) * settings.batchSize + k;
          ret = sets[i]:get(index);
          inputs[k] = ret.inp;
          targets[k] = ret.out;
        end
        
        -- forward pass
        local outputs = modelC:forward(inputs); 
        
        -- frame error evaluation
        for k = 1, settings.batchSize, 1 do
          _, mx = outputs[k]:max(1);

          if (mx:squeeze() ~= targets[k]) then
            err_mx = err_mx + 1;
          end
          all = all + 1;
        end
        
      end
      
      -- save error rate for graph and log
      err = 100 * err_mx / all;
      table.insert(errorTable[i-1], err);
      log.info("Set " .. settings.lists[i] .. " - epoch: " .. epoch .. "/".. settings.noEpochs .. " - err = " .. err);
      
    end

    -- draw frame error rate graph
    if (settings.drawERRs == 1) then
      gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errs.png'); 
      if (#settings.lists-1 == 1) then 
        gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'});
      elseif (#settings.lists-1 == 2) then 
        gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'});
      elseif (#settings.lists-1 == 3) then 
        gnuplot.plot({settings.lists[2], torch.Tensor(errorTable[1]), '-'}, {settings.lists[3], torch.Tensor(errorTable[2]), '-'}, {settings.lists[4], torch.Tensor(errorTable[3]), '-'});
      else
        flog.error('GNUPlot: not supported');
      end    
      gnuplot.title('Error rates');
      gnuplot.xlabel('epoch');
      gnuplot.ylabel('error rate [%]');
      gnuplot.plotflush();
    end
  end
end
