
-- LM --- NN Utils

-- network initialization - literature
function initializeLL(inputSize, outputSize)
  
  local l = nn.Linear(inputSize, outputSize)
  local v = math.sqrt(6.0 / (outputSize + inputSize));
  l.weight = torch.randn(outputSize, inputSize);
  l.weight:mul(v);
  l.bias:zero();
  return l;
    
end

-- pick activation function
function getAF() 
  if (settings.activationFunction == "relu") then
    return nn.ReLU();
  elseif (settings.activationFunction == "tanh") then
    return nn.Tanh();
  elseif (settings.activationFunction == "sigmoid") then
    return nn.Sigmoid();
  else
    error('Activation function: not supported');
  end
end

-- create classic feed forward model
function buildFFModel()
  
  -- input layer
  model = nn.Sequential();
  if (settings.dropout == 1) then model:add(nn.Dropout(settings.dropoutThreshold[1])); end   -- dropout
  model:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons[1]));   -- layer size
  model:add(getAF());   -- layer type
  
  -- hidden layers
  for i = 1, settings.noHiddenLayers do
    if (settings.dropout == 1) then model:add(nn.Dropout(settings.dropoutThreshold[i+1])); end   -- dropout
    model:add(initializeLL(settings.noNeurons[i], settings.noNeurons[i+1]));    -- layer size
    model:add(getAF());   -- layer type
  end
  
  -- output layer
  if (settings.dropout == 1) then model:add(nn.Dropout(settings.dropoutThreshold[settings.dropoutThreshold:size(1)])); end   -- dropout
  ll = nn.Linear(settings.noNeurons[settings.noHiddenLayers+1], settings.outputSize);   -- layer size
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  model:add(nn.LogSoftMax());   -- output layer type
  
  return model;
  
end

-- create classic feed forward model with batch normalization
function buildFFBatchModel()
  
  -- input layer
  model = nn.Sequential();
  
  if (settings.batchInit == 1) then
    model:add(nn.Linear(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons[1]));   -- layer size
  else
    model:add(initializeLL(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons[1]));    -- layer size
  end
  model:add(nn.BatchNormalization(settings.noNeurons[1]));
  model:add(getAF());   -- layer type
  
  -- hidden layers
  for i = 1, settings.noHiddenLayers do
    if (settings.batchInit == 1) then
      model:add(nn.Linear(settings.noNeurons[i], settings.noNeurons[i+1]));    -- layer size
    else
      model:add(initializeLL(settings.noNeurons[i], settings.noNeurons[i+1]));    -- layer size
    end
    model:add(nn.BatchNormalization(settings.noNeurons[i+1]));
    model:add(getAF());   -- layer type
  end
  
  -- output layer
  ll = nn.Linear(settings.noNeurons[settings.noHiddenLayers+1], settings.outputSize);   -- layer size
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  model:add(nn.LogSoftMax());   -- output layer type
  
  return model;
  
end

-- create residual model
function buildResidualModel()
  
  -- input layer
  model = nn.Sequential();
  model:add(nn.Linear(settings.inputSize * (settings.seqL + settings.seqR + 1), settings.noNeurons));   -- layer size
  model:add(nn.BatchNormalization(settings.noNeurons));
  model:add(getAF());   -- layer type

  -- hidden layers
  for i = 1, settings.noHiddenBlocks do   
    mlpTable = nn.ConcatTable();

    mlpA = nn.Sequential();
    for j = 1, settings.blockSize-1, 1 do
      mlpA:add(nn.Linear(settings.noNeurons, settings.noNeurons));
      mlpA:add(nn.BatchNormalization(settings.noNeurons));   
      mlpA:add(getAF());   -- layer type
    end
    mlpA:add(nn.Linear(settings.noNeurons, settings.noNeurons));
    mlpA:add(nn.BatchNormalization(settings.noNeurons)); 
  
    mlpB = nn.Identity();
        
    mlpTable:add(mlpA);
    mlpTable:add(mlpB);   
    
    model:add(mlpTable);
    model:add(nn.CAddTable(true));
    model:add(getAF());   -- layer type
  end

  -- output layer
  ll = nn.Linear(settings.noNeurons, settings.outputSize);   -- layer size
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  model:add(nn.LogSoftMax());   -- output layer type
  
  return model;
  
end
