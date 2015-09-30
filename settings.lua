
require 'torch'
require 'logroll'

-- LM

-- Script settings

function Settings()
    
  local settings = {
    
    -- DNN settings
    inputSize = 39;           -- number of input features
    outputSize = 3886;        -- number of output neurons (= align states)
    --outputSize = 3180;      -- PL
    --outputSize = 2377;      -- SK
    --outputSize = 2183;      -- HR
    --outputSize = 2593;      -- RU 
    noHiddenLayers = 5;     
    noNeurons = torch.Tensor({1024, 1024, 768, 768, 512, 512});       -- size: noHiddenLayers + 1
    noEpochs = 150;
    batchSize = 1024;
    learningRate = 0.08;
    lRDecayActive = 0;        -- 1 active
    learningRateDecay = 0.1;
    dropout = 0;              -- 1 active
    dropoutThreshold = torch.Tensor({0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});  -- size: number of layers
    seqL = 5;                 -- number of frames added as input - left
    seqR = 5;                 -- number of frames added as input - right
    startEpoch = 0;
    borders = 0;              -- 1 active   
    cmsActive = 0;            -- 1 active
    cms = 100;                -- window for calculation of CMS, only for stats&datasetFLMS
    
    -- cuda on/off
    cuda = 1;                 -- 1 active
    
    -- input lists    
    trainFile = 'cz-train-new.list';
    validFile = 'cz-valid-new.list';
    testFile = 'cz-test-new.list';
    
    -- input folders
    inputPath = "/wav/";
    mfccPath = "/fbank39/";
    akuPath = "/akulab/";
    -- input extension
    mfccExt = ".par";
    akuExt = ".akulab";
    
    -- output folders
    outputFolder = "/data/nnModels/2015-09-29";
    statsFolder = "/stats";
    logFolder = "/log";
    modFolder = "/mod";

  }  
  
  -- default global stats
  settings.mean = torch.Tensor(settings.inputSize):zero();    
  settings.var = torch.Tensor(settings.inputSize):fill(1);    
  
  -- log
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/settings.log');
  flog.info(settings);
  
  -- create output folders
  os.execute("mkdir " .. settings.outputFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.statsFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.logFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.modFolder);
  
  return settings;
    
end

