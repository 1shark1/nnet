
require 'torch'
require 'logroll'

function Settings()
    
  local settings = {
    
    inputSize = 39
    --CZ
    outputSize = 3886;
    
    --PL
		--outputSize = 3180;
    
    --SK
    --outputSize = 2377;
    
    --HR
    --outputSize = 2183;
    
    --RU
    --outputSize = 2593;
    
    seqL = 5;
    seqR = 5;
    
    noNeurons = 512;
    noHiddenLayers = 5;
    noEpochs = 150;
    learningRate = 0.08;
    learningRateDecay = 0.1;
    batchSize = 1024;
    startEpoch = 0;
    cms = 100;
    cmsActive = 0;
    dropout = 1;
    
    trainFile = 'cz-train-new.list';
    validFile = 'cz-valid-new.list';
    testFile = 'cz-test-new.list';
    
    outputFolder = "/data/nnModels/2015-05-29";
       
    inputPath = "/wav/";
    mfccPath = "/fbank52/";
    akuPath = "/akulab/";
    mfccExt = ".par";
    akuExt = ".akulab";
    
    statsFolder = "/stats";
    logFolder = "/log";
    modFolder = "/mod";
    
    borders = 0;

  }  
  
  settings.mean = torch.Tensor(settings.inputSize):zero();
  settings.var = torch.Tensor(settings.inputSize):fill(1);
  
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/settings.log');
  flog.info(settings);
  
  os.execute("mkdir " .. settings.outputFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.statsFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.logFolder);
  os.execute("mkdir " .. settings.outputFolder .. settings.modFolder);
  
  return settings;
    
end

