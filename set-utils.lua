
-- LM -- Stats/Dataset Utils -- 5/10/16 --



-- function applying local mean subtraction/CMS to input data
function applyCMS(fvec) 
  
  -- init
  local cms = torch.Tensor(fvec:size(1), settings.inputSize):zero()
  local cmsCounter
  
  -- indices for computation
  local startIndex, endIndex
    
  for i = 1, fvec:size(1), 1 do
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
    if endIndex > fvec:size(1) then
      cmsCounter = cmsCounter - (endIndex - fvec:size(1))
      endIndex = fvec:size(1)
    end
    
    -- compute CMS for given frame   
    cms[{ i, {} }] = (torch.sum(fvec[{ {startIndex, endIndex}, {} }], 1)):div(cmsCounter)
  end
  
  -- apply CMS
  fvec:add(-cms)
  
end  



-- function filling borders of input
function fillBordersInputs(fvec)
  
  -- switch to correct fill of boarders (inputs)
  if settings.cloneBorders == 1 then
    return cloneBordersInputs(fvec)
  elseif settings.cloneBorders == 2 then
    return fillValueBordersInputs(fvec, 0)
  else
    error('FillBordersInputType: not supported')
  end
  
end



-- function cloning borders - inputs
function cloneBordersInputs(fvec) 
  
  local pre = fvec[{{1, {}}}]
  local post = fvec[{{-1, {}}}]
  
  -- clone borders
  if settings.seqL > 0 and settings.seqR > 0 then
    pre = pre:repeatTensor(settings.seqL, 1)
    post = post:repeatTensor(settings.seqR, 1)  
    return torch.cat(pre, torch.cat(fvec, post, 1), 1)
  elseif settings.seqL == 0 and settings.seqR > 0 then
    post = post:repeatTensor(settings.seqR, 1)  
    return torch.cat(fvec, post, 1)
  elseif settings.seqL > 0 and settings.seqR == 0 then
    pre = pre:repeatTensor(settings.seqL, 1)
    return torch.cat(pre, fvec, 1)
  end
  
  return fvec
end



-- function filling borders with number value - inputs
function fillValueBordersInputs(fvec, value) 
  
  local pre, post
  
  -- fill borders
  if settings.seqL > 0 and settings.seqR > 0 then
    pre = torch.Tensor(settings.seqL, settings.inputSize):zero() + value
    post = torch.Tensor(settings.seqR, settings.inputSize):zero() + value
    return torch.cat(pre, torch.cat(fvec, post, 1), 1)
  elseif settings.seqL == 0 and settings.seqR > 0 then
    post = torch.Tensor(settings.seqR, settings.inputSize):zero() + value 
    return torch.cat(fvec, post, 1)
  elseif settings.seqL > 0 and settings.seqR == 0 then
    pre = torch.Tensor(settings.seqL, settings.inputSize):zero() + value
    return torch.cat(pre, fvec, 1)
  end
  
  return fvec

end



-- function filling borders of references
function fillBordersRefs(refs)
  
  -- switch to correct fill of boarders (references)
  if settings.cloneBorders == 1 then
    return cloneBordersRefs(refs)
  elseif settings.cloneBorders == 2 then
    return fillValueBordersRefs(refs, -1)
  else
    error('FillBordersInputType: not supported')
  end
  
end



-- function cloning borders - references
function cloneBordersRefs(refs) 
  
  local curOut = torch.Tensor(refs:size(1) + settings.seqL + settings.seqR)  
  
  for i = 1, settings.seqL, 1 do
    curOut[i] = refs[1]
  end  
  for i = settings.seqL + 1, curOut:size(1) - settings.seqR - settings.seqL, 1 do
    curOut[i] = refs[i]
  end  
  for i = curOut:size(1) - settings.seqR - settings.seqL + 1, curOut:size(1) - settings.seqL, 1 do
    curOut[i] = refs[refs:size(1)]
  end    

  return curOut
  
end



-- function cloning borders - references
function fillValueBordersRefs(refs, val) 
  
  local curOut = torch.Tensor(refs:size(1) + settings.seqL + settings.seqR)  
  
  for i = 1, settings.seqL, 1 do
    curOut[i] = val
  end  
  for i = settings.seqL + 1, curOut:size(1) - settings.seqR - settings.seqL, 1 do
    curOut[i] = refs[i]
  end  
  for i = curOut:size(1) - settings.seqR - settings.seqL + 1, curOut:size(1) - settings.seqL, 1 do
    curOut[i] = val
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


