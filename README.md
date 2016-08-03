# nnet v3.1

version: v3.1 (3/8/2016)

details: package update allowing users to train DNNs without worrying about RAM memory, set settings.packageCount

version: v3 (20/5/2016)

info: a torch based solution to training of dnns (speech recognition, speech activity detection etc.)

inputs: training list, (evaluation lists), input files (htk format), target files (rec-mapped or target frames sequences in binary)

dnn: use settings.lua to configure the dnn parameters

outputs: trained networks (mods, nnets), stats (mean, var, framestats), logs

details: major overhaul + optim support, (40% training speedup, 20% evaluation speedup) over v2

LM

