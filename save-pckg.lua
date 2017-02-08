
-- LM -- Package Save



-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'logroll'

-- require settings
local setts = 'settings-test'
if arg[1] then 
  setts = string.gsub(arg[1], ".lua", "")
end
assert(require(setts))

-- set default tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- initialize settings
settings = Settings()  
  
-- add path to scripts
if settings.scriptFolder then  
  package.path = package.path .. ";" .. settings.scriptFolder .. "?.lua"
end

-- program requires
require 'utils'
require 'io-utils'
require 'set-utils'
require 'dataset'

if settings.savePackage == 1 then
	os.execute("mkdir -p " .. settings.outputFolder .. settings.packageFolder)
	savePackageFilelists(settings.listFolder .. settings.lists[1])

	for i = 1, settings.packageCount, 1 do
		local dataset = Dataset(settings.outputFolder .. settings.packageFolder .. "pckg" .. i .. ".list", true, true, false, true)
		saveFramestats(dataset.framestats, i)
		dataset = nil
		collectgarbage()
	end
end
  