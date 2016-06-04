
-- LM -- DNN Utils -- 4/6/16 --



-- function initializing linear layer
function initializeLL(inputSize, outputSize)
  
  local l = nn.Linear(inputSize, outputSize)
  local v = math.sqrt(6.0 / (outputSize + inputSize))
  l.weight = torch.randn(outputSize, inputSize)
  l.weight:mul(v)
  l.bias:zero()
  return l

end



-- function building the required network
function buildDNN()
  
  if settings.model == "classic" then
    return buildFFModel()
  elseif settings.model == "residual" then
    return buildResidualModel()
  elseif settings.model == "batch" then
    return buildFFModel(true)
  else
    error('Model: not supported')
  end
  
end



-- function building classic feed forwarf DNN model w/ or wo/ batch normalization
function buildFFModel(batch)
  
  -- input layer
  local model = nn.Sequential()
  if settings.dropout == 1 then model:add(nn.Dropout(settings.dropoutThreshold[1])) end                     
  model:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons[1]))    
  if batch then model:add(nn.BatchNormalization(settings.noNeurons[1])) end
  model:add(getAF())    -- transform function
  
  -- hidden layers
  for i = 1, settings.noHiddenLayers do
    if settings.dropout == 1 then model:add(nn.Dropout(settings.dropoutThreshold[i+1])) end   
    model:add(initializeLL(settings.noNeurons[i], settings.noNeurons[i+1]))        
    if batch then model:add(nn.BatchNormalization(settings.noNeurons[i+1])) end
    model:add(getAF())   -- transform function
  end
  
  -- output layer
  if settings.dropout == 1 then model:add(nn.Dropout(settings.dropoutThreshold[settings.dropoutThreshold:size(1)])) end    
  ll = nn.Linear(settings.noNeurons[settings.noHiddenLayers+1], settings.outputSize)                                    
  ll.weight:zero()    
  ll.bias:zero()      
  model:add(ll)
  model:add(getFinalAF())   -- output layer transform function
  
  return model;
  
end



-- function building deep residual network
function buildResidualModel()
  
  -- input layer
  local model = nn.Sequential()
  model:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons))   -- layer size
  model:add(nn.BatchNormalization(settings.noNeurons))
  model:add(getAF())     

  -- hidden layers
  for i = 1, settings.noHiddenBlocks do   
    mlpTable = nn.ConcatTable()

    mlpA = nn.Sequential()
    for j = 1, settings.blockSize-1, 1 do
      mlpA:add(initializeLL(settings.noNeurons, settings.noNeurons))
      mlpA:add(nn.BatchNormalization(settings.noNeurons))
      mlpA:add(getAF())   
    end
    mlpA:add(nn.initializeLL(settings.noNeurons, settings.noNeurons))
    mlpA:add(nn.BatchNormalization(settings.noNeurons))
  
    mlpB = nn.Identity()
  
    mlpTable:add(mlpA)
    mlpTable:add(mlpB)

    model:add(mlpTable)
    model:add(nn.CAddTable(true))
    model:add(getAF())
  end

  -- output layer
  ll = nn.Linear(settings.noNeurons, settings.outputSize)
  ll.weight:zero()
  ll.bias:zero()
  model:add(ll)
  model:add(getFinalAF())
  
  return model
  
end



-- function selecting transfer function
function getAF() 
  
  if settings.activationFunction == "relu" then
    return nn.ReLU()
  elseif settings.activationFunction == "tanh" then
    return nn.Tanh()
  elseif settings.activationFunction == "sigmoid" then
    return nn.Sigmoid()
  else
    error('Activation function: not supported')
  end
  
end



-- function selecting transfer function for last layer
function getFinalAF() 
  
  if settings.finalActivationFunction == "logsoftmax" then
    return nn.LogSoftMax()
  else
    error('Final activation function: not supported')
  end
  
end



-- function selecting transfer function for last layer
function getCriterion() 
  
  if settings.criterion == "nll" then
    return nn.ClassNLLCriterion()
  else
    error('Criterion: not supported')
  end
  
end



-- function selecting correct settings for optim
function getOptimParams()
  
  if settings.optimization == "sgd" then
    return getOptimParamsSGD()
  elseif settings.optimization == "other" then
    return getOptimParamsOther()
  else
    error('Optimization: not supported')
  end
  
end



-- function preparing settings for optim - SGD
function getOptimParamsSGD()
  
  local state
  
  state = {
    learningRate = settings.learningRate,   
    learningRateDecay = settings.learningRateDecay,
    momentum = settings.momentum  
  }
  
  return state
  
end



-- function preparing settings for optim - other (testing purposes)
function getOptimParamsOther()
  
  local state, config
  
  state = {}
  config = {}
  
  return state, config
  
end


