import os
import shutil

wav_container= "C:\\TESIS-OLD\\tesis-data-base\\waveform_names_rcc_only_replicas_20160117_1mes.out"
destiny = 'C:\\TESIS-OLD\\datos\\wav_rcc_only_20160117_1mes'
path_to_find= "D:\\wav2010-2023\\2016\\"

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)

with open(wav_container, "r") as fname:
	files = fname.readlines()
linea = 0
not_found = 0
for file in files:    
    origin = findfile(file.strip('\n'), path_to_find)
    if origin is not None:
        shutil.copy(origin, destiny)
        linea = linea +1
        print(linea)
    else:
        not_found= not_found +1
print('No encontrados')
print(not_found)