
-- LM -- DNN Decoding -- 11/8/16 --



-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'logroll'
require 'nn'

-- require settings
local setts = 'settings-test'
if arg[1] then 
  setts = string.gsub(arg[1], ".lua", "")
end
assert(require(setts))

-- set default tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- initialize settings
settings = Settings(true) 

-- add path to scripts
if settings.scriptFolder then  
  package.path = package.path .. ";" .. settings.scriptFolder .. "?.lua"
end

-- program requires
require 'utils'
require 'io-utils'
require 'set-utils'
require 'nn-utils'
require 'dataset'

-- require cuda
if settings.cuda == 1 then
  require 'cunn'
  require 'cutorch'  
end

-- load model from file
if not paths.filep(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod") then  
  error('File ' .. settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod" .. ' does not exist!')
end
local model = torch.load(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod")

-- load stats
settings.mean, settings.std = readStats() 

-- load framestats
local framestats, count
if settings.applyFramestats == 1 then
  framestats, count = readFramestats(settings.outputFolder .. settings.statsFolder .. '/framestats.list')
end

-- cuda on/off
if settings.cuda == 1 then 
  model:cuda()
end

-- initialize logs
local flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/decode.log')
local plog = logroll.print_logger()
local log = logroll.combine(flog, plog)

-- evaluation mode
model:evaluate()

-- load filelist
local filelist = readFilelist(settings.listFolder .. settings.decodeFile)

-- decode files one by one
for file = 1, #filelist, 1 do   
  
  -- prepare file
  local dataset = Dataset(filelist[file], false, false, false, false, true)
  log.info("Decoding file: " .. filelist[file] .. " type: " .. settings.decodeType)
  
  -- compute number of batches
  local noBatches = math.ceil(dataset:size() / settings.batchSize)
  
  -- prepare output folder structure
  local folders = split(filelist[file], "/")
  local fileParts = split(folders[#folders], ".")
  os.execute("mkdir -p " .. settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1])
  
  -- open according file
  local ff
  if settings.decodeType == "lkl" then
    ff = torch.DiskFile(settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1] .. "/" .. fileParts[1] .. "." .. settings.decodeType, "w")
    ff:binary()
    saveHTKHeader(ff, dataset:size(), true)
  elseif settings.decodeType == "txt" or settings.decodeType == "csv" then
    ff = io.open(settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1] .. "/" .. fileParts[1] .. "." .. settings.decodeType, "w")
  else
    error('DecodeType: not supported')
  end

  -- process batches
  for noBatch = 1, noBatches, 1 do   
    
    -- last batch fix
    local batchSize = settings.batchSize
    if noBatch == noBatches then
      batchSize = dataset:size() - ((noBatches-1) * settings.batchSize)
    end
  
    -- prepare input tensor
    local inputs = torch.Tensor(batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero()
  
    -- process batches
    for i = 1, batchSize, 1 do
      local index = (noBatch - 1) * settings.batchSize + i
      local ret = dataset[index]
      inputs[i] = ret.inp
    end
    
    -- cuda neccessities
    if settings.cuda == 1 then
      inputs = inputs:cuda()
    end     
    
    -- feed forward
    local pred = model:forward(inputs)  
    
    if settings.cuda == 1 then
      pred = pred:typeAs(settings.mean)
    end     
    
    -- normalize using framestats
    if settings.applyFramestats == 1 then
      pred = applyFramestats(pred, framestats, count)
    end
    
    -- save output
    if settings.decodeType == "lkl" then
      ff:writeFloat(pred:storage())
    elseif settings.decodeType == "txt" then
      for i = 1, batchSize, 1 do
        local _, mx = pred[i]:max(1)
        ff:write(mx[1]-1 .. "\n")
      end
    elseif settings.decodeType == "csv" then
      for i = 1, batchSize, 1 do
        pred[i] = pred[i]:exp()
        for j = 1, pred[i]:size(1), 1 do
          ff:write(pred[i][j])
          if j ~= pred[i]:size(1) then
            ff:write(";")
          end
        end
        ff:write("\n")
      end
    end 
  end  

  ff:close()
  
end


