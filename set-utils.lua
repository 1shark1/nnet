
-- LM -- Stats/Dataset Utils -- 20/5/16 --



-- function applying local mean subtraction/CMS to input data
function applyCMS(fvec, nSamples) 
  
  -- init
  local cms = torch.Tensor(nSamples, settings.inputSize):zero()
  local cmsCounter
  
  -- indices for computation
  local startIndex, endIndex
    
  for i = 1, nSamples, 1 do
    -- find indices
    cmsCounter = settings.cmsSize + 1
    startIndex = i - (settings.cmsSize / 2)
    endIndex = i + (settings.cmsSize / 2)
    
    -- beginning exception
    if startIndex < 1 then
      cmsCounter = cmsCounter + startIndex - 1
      startIndex = 1
    end
    
    -- ending exception
    if endIndex > nSamples then
      cmsCounter = cmsCounter - (endIndex - nSamples)
      endIndex = nSamples
    end
    
    -- compute CMS for given frame   
    cms[{ i, {} }] = (torch.sum(fvec[{ {startIndex, endIndex}, {} }], 1)):div(cmsCounter)
  end
  
  -- apply CMS
  fvec:add(-cms)
  
end  



-- function cloning borders - inputs
function cloneBordersInputs(data, fvec) 
  
  local pre = torch.Tensor(data, 1, torch.LongStorage{1, settings.inputSize})
  local post = torch.Tensor(data, (fvec:size(1) - 1) * settings.inputSize + 1, torch.LongStorage{1, settings.inputSize})  
  
  pre = pre:repeatTensor(settings.seqL, 1)
  post = post:repeatTensor(settings.seqR, 1)  
  
  fvec = torch.cat(pre, torch.cat(fvec, post, 1), 1)
  
end



-- function cloning borders - targets
function cloneBordersRefs(data) 
  
  local curOut = torch.Tensor(data:size(1) + settings.seqL + settings.seqR)  
  
  for i = 1, settings.seqL, 1 do
    curOut[i] = data[1]
  end  
  for i = settings.seqL + 1, curOut:size(1) - settings.seqR - settings.seqL, 1 do
    curOut[i] = data[i]
  end  
  for i = curOut:size(1) - settings.seqR - settings.seqL + 1, curOut:size(1) - settings.seqL, 1 do
    curOut[i] = data[data:size(1)]
  end    

  return curOut
  
end



-- function applying framestats -- inputs -/+ (ln(framestats) / ln(count))
function applyFramestats(inputs, framestats, count)
  
  local fstats = framestats:clone()
  local pred

  fstats = fstats:repeatTensor(inputs:size(1), 1)
  
  if settings.applyFramestatsType == 0 then
    pred = inputs - (fstats / count):log()
  elseif settings.applyFramestatsType == 1 then 
    pred = inputs + (fstats / count):log()
  else
    error('Operation: not supported')
  end
  
  return pred
  
end


