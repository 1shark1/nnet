
-- LM -- Settings -- 20/5/16 --



function Settings(decode)
  
  local settings = {}
  
  -- general DNN settings
  settings.inputSize = 39
  settings.outputSize = 3875  
  settings.noEpochs = 10
  settings.startEpoch = 0
  settings.batchSize = 1024
  settings.seqL = 5                                 -- input context window - number of preceding frames
  settings.seqR = 5                                 -- input context window - number of following frames
  settings.learningRate = 0.08
  settings.learningRateDecay = 0
  settings.momentum = 0
  settings.computeStats = 1                         -- compute global normalizaion stats, on/off
  if settings.computeStats == 0 then
    settings.mean = torch.Tensor(settings.inputSize):zero()  
    settings.std = torch.Tensor(settings.inputSize):fill(1)   
  end
  settings.applyCMS = 0                             -- compute & apply local mean normalization on/off
  if settings.applyCMS == 1 then
    settings.cmsSize = 100
  end  
  settings.cloneBorders = 0                         -- clone first & last frame to allow training on border frames, on/off
  settings.dnnAlign = 0                             -- ntx4 dnn align, on/off
  settings.activationFunction = "relu"              -- relu / tanh / sigmoid
  settings.finalActivationFunction = "logsoftmax"   -- logsoftmax
  settings.criterion = "nll"                        -- nll
  settings.optimization = "sgd"                     -- sgd
  
  -- model based settings
  settings.model = "classic"                        -- classic / residual / batch
  if settings.model == "classic" or settings.model == 'batch' then
    settings.noHiddenLayers = 5
    settings.noNeurons = torch.Tensor({768, 768, 768, 768, 768, 768})                 -- size: noHiddenLayers + 1
    settings.dropout = 0             
    if settings.dropout == 1 then
      settings.dropoutThreshold = torch.Tensor({0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1})   -- size: number of layers
    end
  elseif settings.model == "residual" then  
    settings.noHiddenBlocks = 25
    settings.blockSize = 2
    settings.noNeurons = 768
  end
  
  -- other settings
  settings.cuda = 1
  settings.shuffle = 1                              -- shuffle training data, on/off
  settings.exportNNET = 1
  settings.drawERRs = 1
  settings.inputView = 0                            -- read view input files, on/off
  settings.inputType = "htk"                        -- htk
  settings.refType = "rec-mapped"                   -- akulab / rec-mapped
  
  -- path settings
  settings.sameFolder = 1                           -- 0 original solution (folders: settings -> input folders); 1 in one directory
  if settings.sameFolder == 0 then
    settings.parPath = "/fbank39/"
    settings.refPath = "/rec.mapped/"
  end  
  settings.parExt = ".fbc3916"
  settings.refExt = ".rec.mapped"
  settings.listFolder = "lists/"
  settings.lists = {'cz-cz48-train.list', 'cz-cz48-test.list', 'cz-cz48-valid.list'}    -- lists, first for training, the rest for evaluation
  settings.modelName = "moje-sit"
  settings.outputFolder = "/data/nnModels/" .. settings.modelName
  settings.statsFolder = "/stats/"
  settings.logFolder = "/log/"
  settings.modFolder = "/mod/"
  settings.logPath = "settings.log"

  -- decode settings 
  if decode then
    -- main decode settings
    settings.decodeFile = 'cz-test-decode-small.list'
    settings.decodeFolder = '/decoded/'
    settings.decodeType = 'txt'                     -- lkl / txt
    settings.parExt = ".fbc3916"
    settings.startEpoch = 1
    settings.batchSize = 1024
    settings.modelName = "moje-sit"
    settings.outputFolder = "/data/nnModels/" .. settings.modelName
    settings.applyFramestats = 1                    -- apply a priori probability to DNN outputs
    if settings.applyFramestats == 1 then
      settings.applyFramestatsType = 0              -- 0 (-) recommended / 1 (+)
    end
    
    -- other decode settings
    settings.logPath = "decode.log"
    os.execute("mkdir -p " .. settings.outputFolder .. settings.decodeFolder)
  end

  -- log
  local flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. settings.logPath)
  flog.info(settings)
  
  -- create output folders
  os.execute("mkdir -p " .. settings.outputFolder .. settings.statsFolder)
  os.execute("mkdir -p " .. settings.outputFolder .. settings.modFolder)
  
  return settings
  
end


