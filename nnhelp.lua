
require 'torch'
require 'nn'

function initializeLL(inputSize, outputSize)
  
  local l = nn.Linear(inputSize, outputSize)
  local v = math.sqrt(6.0 / (outputSize + inputSize));
  l.weight = torch.randn(outputSize, inputSize);
  l.weight:mul(v);
  l.bias:zero();
  return l;
    
end

function exportModel(ifile, ofile)
  
  local mlp = torch.load(ifile);
  local linearNodes = mlp:findModules('nn.Linear');
  local f = torch.DiskFile(ofile, "w");
  f:binary();
  f:writeInt(0x54454E4e);
  f:writeInt(0);
  local isize = linearNodes[1].weight:size(2);
  f:writeInt(isize);
  local noLayers = #linearNodes;
  f:writeInt(noLayers); 
  for i = 1, noLayers do
    -- m, n = linearNodes[i].weight:size();
    local noNeurons = linearNodes[i].weight:size(1);
    f:writeInt(noNeurons);
  end
  for i = 1, noLayers do
    local stor = linearNodes[i].bias:float():storage();
    f:writeFloat(stor);
  end      
  for i = 1, noLayers do
    local stor = linearNodes[i].weight:float():storage();
    f:writeFloat(stor);
  end
  f:close();
  
end