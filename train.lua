
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'cunn'
require 'cutorch'
require 'logroll'
require 'gnuplot'

require 'nnhelp'
require 'settings'

--require 'exportLocal'
--require 'exportLocalLCMS'
--require 'exportLocalLCMSV'
require 'exportFLocal'
--require 'exportFLocalV'

--require 'stats'
--require 'statsCMS'
--require 'statsLCMS'
--require 'statsFLMS'
--require 'statsFLMSV'
require 'statsFLMSBorders'

--require 'dataset'
--require 'datasetBorders'
--require 'datasetCMSSlow'
--require 'datasetCMS'
--require 'datasetLCMSSlow'
--require 'datasetLCMS'
--require 'datasetFLMS'
--require 'datasetFLMSV'
require 'datasetFLMSBorders'

settings = Settings();

export = ExportLocal("/data/testData/list", settings);
error();

stats = Stats(settings.trainFile, settings);
stats:exportStats();

settings.CMSmean = stats.CMSmean;
settings.CMSvar = stats.CMSvar;

settings.mean = stats.mean;
settings.var = stats.var;
dataset = Dataset(settings.trainFile, settings);
datasetValid = Dataset(settings.validFile, settings);
datasetTest = Dataset(settings.testFile, settings);


flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

noBatches = (dataset:size() - dataset:size() % settings.batchSize) / settings.batchSize;
noBatchesValid = (datasetValid:size() - datasetValid:size() % settings.batchSize) / settings.batchSize;
noBatchesTest = (datasetTest:size() - datasetTest:size() % settings.batchSize) / settings.batchSize;

if settings.startEpoch == 0 then        
  mlp = nn.Sequential();
  if (settings.dropout == 1) then mlp:add(nn.Dropout); end
  mlp:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1 + settings.cmsActive), settings.noNeurons));
  mlp:add(nn.ReLU());
  
  for i = 1, settings.noHiddenLayers do
    if (settings.dropout == 1) then mlp:add(nn.Dropout); end
    mlp:add(initializeLL(settings.noNeurons, settings.noNeurons));
    mlp:add(nn.ReLU());
  end
  
  if (settings.dropout == 1) then mlp:add(nn.Dropout); end
  ll = nn.Linear(settings.noNeurons, settings.outputSize);  
  ll.weight:zero();
  ll.bias:zero();
  mlp:add(ll);
  mlp:add(nn.LogSoftMax());
else
  if paths.filep(settings.outputFolder .. "/mod/" .. settings.startEpoch .. ".mod") then  
    mlp = torch.load(settings.outputFolder .. "/mod/" .. settings.startEpoch .. ".mod");
    log.info('Epoch  ' .. settings.startEpoch - 1 .. ' loaded');
  else
    log.error('Epoch ' .. settings.startEpoch .. ".mod" .. ' does not exist!');
  end
end

mlp:cuda();

mlp_auto = nn.Sequential();
mlp_auto:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));
mlp_auto:add(mlp);
mlp_auto:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')); 


local validError = {};
local testError = {};

for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do
     
  shuffle = torch.randperm(dataset:size(), 'torch.LongTensor');
  mlp_auto:training();
  log.info("Training epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  local etime = sys.clock();
  
  for noBatch = 1, noBatches, 1 do

    --plog.info("Epoch: " .. epoch .. ", Batch: " .. noBatch .. "/" .. noBatches);       
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1 + settings.cmsActive)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
        
    local btime = sys.clock();  
    
    for i = 1, settings.batchSize, 1 do
      
      local index = (noBatch - 1) * settings.batchSize + i;     
      if(shuffle) then
        index = shuffle[index];
      end
            
      ret = dataset:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;

    end
  
    --local lr = settings.learningRate / (1 + (epoch - 1) * settings.learningRateDecay);
    
    local criterion = nn.ClassNLLCriterion();
    mlp_auto:zeroGradParameters();
  
    local pred = mlp_auto:forward(inputs);
  
    local t = criterion:backward(pred, targets);
    mlp_auto:backward(inputs, t);
    --mlp_auto:updateParameters(lr);
    mlp_auto:updateParameters(settings.learningRate);
    
  end
  
  plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs);
  torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", mlp);
  exportModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet");
  
  log.info("Epoch: " .. epoch .. "/".. settings.noEpochs .. " completed in " .. sys.clock() - etime);
  
  plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs);
  
  --valid
      
  err_mx = 0;
  all = 0;

  mlp_auto:evaluate();

  for noBatchValid = 1, noBatchesValid, 1 do
    
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1 + settings.cmsActive)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    for i = 1, settings.batchSize, 1 do
      local index = (noBatchValid - 1) * settings.batchSize + i;
      ret = datasetValid:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;
    end
    
    local outputs = mlp_auto:forward(inputs); 
        
    for i = 1, settings.batchSize, 1 do
      _, mx = outputs[i]:max(1);

      if mx:squeeze() ~= targets[i] then
        err_mx = err_mx + 1;
      end
      all = all + 1;
    end
  end

  table.insert(validError, 100 * err_mx / all);
  log.info("Validation set - epoch: " .. epoch .. "/".. settings.noEpochs .. " - errmx = " .. 100 * err_mx / all);
  
  --test
  
  err_mx = 0;
  all = 0;

  for noBatchTest = 1, noBatchesTest, 1 do
    
    local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1 + settings.cmsActive)):zero();
    local targets = torch.Tensor(settings.batchSize):zero();
    
    for i = 1, settings.batchSize, 1 do
      local index = (noBatchTest - 1) * settings.batchSize + i;
      ret = datasetTest:get(index);
      inputs[i] = ret.inp;
      targets[i] = ret.out;
    end
    
    local outputs = mlp_auto:forward(inputs);    
    
    for i = 1, settings.batchSize, 1 do
      _, mx = outputs[i]:max(1);

      if mx:squeeze() ~= targets[i] then
        err_mx = err_mx + 1;
      end
      all = all + 1;
    end
  end

  table.insert(testError, 100 * err_mx / all);
  log.info("Test set - epoch: " .. epoch .. "/".. settings.noEpochs .. " - errmx = " .. 100 * err_mx / all);
  
  gnuplot.pngfigure(settings.outputFolder .. settings.statsFolder .. '/errs.png');
  gnuplot.plot({'Validation set', torch.Tensor(validError), '-'}, {'Test set', torch.Tensor(testError), '-'});
  gnuplot.title('Valid and Test Error rates');
  gnuplot.xlabel('epoch');
  gnuplot.ylabel('error rate [%]');
  gnuplot.plotflush();
  
end

