
-- LM --- Settings

function Settings(decode)
    
  local settings = {};
  
  -- general DNN settings
  settings.inputSize = 39;
  settings.outputSize = 3886;  
  settings.noEpochs = 15;
  settings.startEpoch = 0;
  settings.batchSize = 1024;
  settings.seqL = 5;                  -- number of frames added as input - left
  settings.seqR = 5;                  -- number of frames added as input - right
  settings.learningRate = 0.08;
  settings.lrDecayActive = 0;
  if(settings.lrDecayActive == 1) then
    settings.lrDecay = 0.1;
  end  
  settings.computeStats = 1;
  if (settings.computeStats == 0) then
    settings.mean = torch.FloatTensor(settings.inputSize):zero();    
    settings.var = torch.FloatTensor(settings.inputSize):fill(1);   
  end
  settings.exportFramestats = 1;
  settings.applyCMS = 1;
  if(settings.applyCMS == 1) then
    settings.cmsSize = 100;
  end  
  settings.cloneBorders = 0;
  settings.dnnAlign = 0;
  settings.activationFunction = "relu";      -- relu / tanh / sigmoid
  
  -- model based settings
  settings.model = "classic"      -- classic / residual / batch
  if (settings.model == "classic") then
    settings.noHiddenLayers = 5;
    settings.noNeurons = torch.Tensor({512, 512, 512, 512, 512, 512});        -- size: noHiddenLayers + 1
    settings.dropout = 0;             
    if (settings.dropout == 1) then
      settings.dropoutThreshold = torch.Tensor({0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});  -- size: number of layers
    end
  elseif (settings.model == "residual") then  
    settings.noHiddenBlocks = 10;
    settings.blockSize = 2;
    settings.noNeurons = 512;
  elseif (settings.model == "batch") then
    settings.noHiddenLayers = 5;
    settings.noNeurons = torch.Tensor({768, 768, 768, 768, 768, 768});        -- size: noHiddenLayers + 1
  end
  
  -- other settings
  settings.cuda = 1;
  settings.shuffle = 1;
  settings.exportNNET = 1;
  settings.drawERRs = 1;
  settings.inputView = 0;               -- filetype: view; for context data
  settings.inputType = "htk";           -- htk
  settings.refType = "rec-mapped"           -- akulab / rec-mapped
  
  -- path settings
  settings.sameFolder = 0;    -- 0 original solution (folders: settings -> input folders); 1 in one directory
  if (settings.sameFolder == 0) then
    settings.parPath = "/fbank39/";
    settings.refPath = "/rec-mapped/";
  end  
  settings.parExt = ".par";
  settings.refExt = ".rec.mapped";
  settings.listFolder = "lists/"; 
  settings.lists = {'cz-train.list', 'cz-test.list', 'cz-valid.list'};
  settings.modelName = "2016-03-batch-hl5-n768";
  settings.outputFolder = "/data/nnModels/" .. settings.modelName;
  settings.statsFolder = "/stats/";
  settings.logFolder = "/log/";
  settings.modFolder = "/mod/";
  settings.logPath = "settings.log";

  -- decode settings 
  if (decode) then
    -- main decode settings
    settings.decodeFile = 'cz-decode';
    settings.decodeFolder = '/decoded/';
    settings.decodeType = 'lkl'           -- lkl / txt
    settings.startEpoch = 10;
    settings.applyFramestats = 1;
    if (settings.applyFramestats == 1) then
      settings.applyFramestatsType = 0;     -- 0 (-) / 1 (+)
    end
    
    -- other decode settings
    settings.logPath = "decode.log";
    os.execute("mkdir -p " .. settings.outputFolder .. settings.decodeFolder);
  end

  -- log
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. settings.logPath);
  flog.info(settings);
  
  -- create output folders
  os.execute("mkdir -p " .. settings.outputFolder .. settings.statsFolder);
  os.execute("mkdir -p " .. settings.outputFolder .. settings.modFolder);
  
  return settings;
    
end
