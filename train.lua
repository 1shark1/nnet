
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'

-- LM

-- Main training script

require 'nnhelp'
require 'settings'

-- initialize settings
settings = Settings();    

-- load modules for cuda if selected
if (settings.cuda == 1) then
  require 'cunn'
  require 'cutorch'  
end

-- load appropriate stats & dataset classes
if (settings.cmsActive == 1) then
  require 'statsFLMSBorders'
  require 'datasetFLMSBorders'
else
  require 'stats'
  require 'dataset'
end

-- compute and export stats of train files
stats = Stats(settings.trainFile, settings);  
stats:exportStats();    

-- override default stats by ones computed on train data
settings.mean = stats.mean;
settings.var = stats.var;
if (settings.cmsActive == 1) then
  settings.CMSmean = stats.CMSmean;   
  settings.CMSvar = stats.CMSvar;
end

-- prepare datasets
dataset = Dataset(settings.trainFile, settings);
datasetValid = Dataset(settings.validFile, settings);
datasetTest = Dataset(settings.testFile, settings);

-- initialize logs
flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

-- compute number of batches
noBatches = (dataset:size() - dataset:size() % settings.batchSize) / settings.batchSize;
noBatchesValid = (datasetValid:size() - datasetValid:size() % settings.batchSize) / settings.batchSize;
noBatchesTest = (datasetTest:size() - datasetTest:size() % settings.batchSize) / settings.batchSize;

-- initialize the network
if settings.startEpoch == 0 then     
  -- input layer
  mlp = nn.Sequential();
  if (settings.dropout == 1) then mlp:add(nn.Dropout(dropoutThreshold[1])); end   -- dropout
  mlp:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons[1]));   -- layer size
  mlp:add(nn.ReLU());   -- layer type
  
  -- hidden layers
  for i = 1, settings.noHiddenLayers do
    if (settings.dropout == 1) then mlp:add(nn.Dropout(dropoutThreshold[i+1])); end   -- dropout
    mlp:add(initializeLL(settings.noNeurons[i], settings.noNeurons[i+1]));    -- layer size
    mlp:add(nn.ReLU());   -- layer type
  end
  
  -- output layer
  if (settings.dropout == 1) then mlp:add(nn.Dropout(dropoutThreshold[settings.dropoutThreshold:size(1)])); end   -- dropout
  ll = nn.Linear(settings.noNeurons[settings.noHiddenLayers+1], settings.outputSize);   -- layer size
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  mlp:add(ll);
  mlp:add(nn.LogSoftMax());   -- output layer type
  
-- load epoch to continue training 
else
  if paths.filep(settings.outputFolder .. "/mod/" .. settings.startEpoch .. ".mod") then  
    mlp = torch.load(settings.outputFolder .. "/mod/" .. settings.startEpoch .. ".mod");
    log.info('Epoch  ' .. settings.startEpoch .. ' loaded');
  else
    log.error('Epoch ' .. settings.startEpoch .. ".mod" .. ' does not exist!');
  end
end

-- cuda on/off
if (settings.cuda == 1) then 
  mlp:cuda();
  mlp_auto = nn.Sequential();
  mlp_auto:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));
  mlp_auto:add(mlp);
  mlp_auto:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'));    
  flog.info("Using CUDA"); 
else
  mlp_auto = mlp;   
  flog.info("Using CPU");
end

-- tensors for frame error rates
local validError = {};
local testError = {};

-- training
for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do
  
  -- timer per epoch - start
  local etime = sys.clock();   
  
  -- shuffle data
  shuffle = torch.randperm(dataset:size(), 'torch.LongTensor');
  
  -- dropout active: mode training
  if (settings.dropout == 1) then mlp_auto:training(); end
  
  -- log
  log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  -- training per batches
  for noBatch = 1, noBatches, 1 do

    -- prepare inputs & outputs tensors    
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    -- process batches
    for i = 1, settings.batchSize, 1 do
      
      -- pick frame (shuffled)
      local index = (noBatch - 1) * settings.batchSize + i;     
      if(shuffle) then
        index = shuffle[index];
      end
      
      -- get data for picked frame and fill input arrays for network
      ret = dataset:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;

    end
  
    -- learning rate decay
    if (settings.lRDecayActive == 1) then local lr = settings.learningRate / (1 + (epoch - 1) * settings.learningRateDecay); end

    -- criterion 
    local criterion = nn.ClassNLLCriterion();
  
    -- forward propagation
    local pred = mlp_auto:forward(inputs);
    criterion:forward(pred, targets);
	
    -- zero the accumulation of the gradients
    mlp_auto:zeroGradParameters();
  
    -- back propagation
    local t = criterion:backward(pred, targets);
    mlp_auto:backward(inputs, t);
    
    -- update parameters
    if (settings.lRDecayActive == 1) then mlp_auto:updateParameters(lr);
    else mlp_auto:updateParameters(settings.learningRate);  
    end
    
  end
  
  -- logs & export model
  plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs);
  torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", mlp);
  exportModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet");
  
  log.info("Epoch: " .. epoch .. "/".. settings.noEpochs .. " completed in " .. sys.clock() - etime);  
  plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  
  -- validation
  err_mx = 0;
  all = 0;

  -- dropout active: mode evaluation
  if (settings.dropout == 1) then mlp_auto:evaluate(); end
  
  -- evaluation per batches
  for noBatchValid = 1, noBatchesValid, 1 do
    
    -- prepare inputs & outputs tensors    
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    -- process batches
    for i = 1, settings.batchSize, 1 do
      -- pick frame and obtain data
      local index = (noBatchValid - 1) * settings.batchSize + i;
      ret = datasetValid:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;
    end
    
    -- forward pass
    local outputs = mlp_auto:forward(inputs); 
    
    -- frame error evaluation
    for i = 1, settings.batchSize, 1 do
      _, mx = outputs[i]:max(1);

      if mx:squeeze() ~= targets[i] then
        err_mx = err_mx + 1;
      end
      all = all + 1;
    end
    
  end

  -- save error rate for graph and log
  table.insert(validError, 100 * err_mx / all);
  log.info("Validation set - epoch: " .. epoch .. "/".. settings.noEpochs .. " - errmx = " .. 100 * err_mx / all);
  
  
  -- test
  err_mx = 0;
  all = 0;
  
  -- dropout active: mode evaluation
  if (settings.dropout == 1) then mlp_auto:evaluate(); end

  -- evaluation per batches
  for noBatchTest = 1, noBatchesTest, 1 do
    
    -- prepare inputs & outputs tensors   
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    -- process batches
    for i = 1, settings.batchSize, 1 do
      -- pick frame and obtain data
      local index = (noBatchTest - 1) * settings.batchSize + i;
      ret = datasetTest:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;
    end
    
    -- forward pass
    local outputs = mlp_auto:forward(inputs);    
    
    -- frame error evaluation
    for i = 1, settings.batchSize, 1 do
      _, mx = outputs[i]:max(1);

      if mx:squeeze() ~= targets[i] then
        err_mx = err_mx + 1;
      end
      all = all + 1;
    end
  end
  
  -- save error rate for graph and log
  table.insert(testError, 100 * err_mx / all);
  log.info("Test set - epoch: " .. epoch .. "/".. settings.noEpochs .. " - errmx = " .. 100 * err_mx / all);
  
  
  -- draw frame error rate graph
  gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errs.png');
  gnuplot.plot({'Validation set', torch.Tensor(validError), '-'}, {'Test set', torch.Tensor(testError), '-'});
  gnuplot.title('Valid and Test Error rates');
  gnuplot.xlabel('epoch');
  gnuplot.ylabel('error rate [%]');
  gnuplot.plotflush();
  
end

