from obspy import read, read_inventory
import  obspy.io.nordic.core as nd
import pandas as pd
import string, os, sys
import matplotlib.pyplot as plt
import numpy as np
import math


dir_rea = 'C:\\TESIS-OLD\\datos\\rea'
dir_wav = 'C:\\TESIS-OLD\\datos\\wav'
dir_output = 'C:\\TESIS-OLD\\datos'
files_rea = os.listdir(dir_rea)
inv_Z = read_inventory('C:\\TESIS-OLD\\datos\\RESP.CW.RCC.00.HHZ')
inv_N = read_inventory('C:\\TESIS-OLD\\datos\\RESP.CW.RCC.00.HHN')
inv_E = read_inventory('C:\\TESIS-OLD\\datos\\RESP.CW.RCC.00.HHE')
time_before = 1
time_after = 4
freq_min_filter = 2
freq_max_filter = 15
sampling_rate_defined = 100
percentile = 95
rea_files_with_waveforms_and_exist_waveform = 0
rea_files_cuantity = 0
rea_files_with_waveforms = 0
cuantity_containing_P = 0
cuantity_containing_S = 0
sampling_rate_100 = 0
snr_s_windows_ok = 0
snr_p_windows_ok = 0
added_to_dataframe = 0

def snr(signal,noise):
    signal_sorted = np.sort(signal)
    noise_sorted = np.sort(noise)
    signal_percentil_95 = signal_sorted[:95]
    noise_percentil_95 = noise_sorted[:95]
    signal_norm_2 = np.linalg.norm(signal_percentil_95)
    noise_norm_2 = np.linalg.norm(noise_percentil_95)
    
    snr = 10 * math.log10(signal_norm_2 ** 2 / noise_norm_2 ** 2)
    return snr


def snr_2(signal,noise):
    signal_sorted = np.sort(signal)
    noise_sorted = np.sort(noise)    
    snr = 10 * math.log10(signal_sorted[95] ** 2 / noise_sorted[95] ** 2)
    return snr


def calculate_azimut(back_azimuth):
    if back_azimuth >= 0 and back_azimuth <= 180:
        azimuth = back_azimuth + 180
    else:
        azimuth = back_azimuth - 180
    return azimuth
    



df = pd.DataFrame(columns=['Origin_Time',
                           'Latitude',
                           'Longitude',
                           'Depth',
                           'RMS',
                           'ML',
                           'Mc',
                           'MW',
                           'P_time',
                           'S_time',
                           'Distance',
                           'Back_Azimuth',
                           'Azimuth',
                           'WAV_file',
                           'RCC_OBSPY',
                           'Z_data',
                           'N_data',
                           'E_data',
                           'SNR_Z_S',
                           'SNR_N_S',
                           'SNR_E_S',
                           'SNR_Z_P',
                           'SNR_N_P',
                           'SNR_E_P',                           
                           'SNR_2_Z_S',
                           'SNR_2_N_S',
                           'SNR_2_E_S',
                           'SNR_2_Z_P',
                           'SNR_2_N_P',
                           'SNR_2_E_P'
                           ])

def search_files(filenames, path):
    for filename in filenames:
        if  os.path.exists(path + filename) != True:
            return False
    return True

for f in files_rea: 
    rea_files_cuantity = rea_files_cuantity +1
    contain_P = False
    snr_Z_S =None
    snr_N_S =None
    snr_E_S =None
    snr_Z_P =None
    snr_N_P =None
    snr_E_P =None
    snr_2_Z_S =None
    snr_2_N_S =None
    snr_2_E_S =None
    snr_2_Z_P =None
    snr_2_N_P =None
    snr_2_E_P =None
    s_time_arrival = None
    contain_Azimuth_Dist= False
    print ('Archivo REA :',dir_rea + os.sep + f)
    waveForms= nd.readwavename(dir_rea + os.sep + f)
    print('Cantidad de waveform :',len(waveForms))
    print (waveForms)

    #---------------Ejecuto solo si el evento contiene los waveforms ------------
    if len(waveForms) != 0 :
        rea_files_with_waveforms = rea_files_with_waveforms +1
        exists_files = search_files(waveForms,dir_wav + os.sep)
        if exists_files == True :
            print('Existen las trazas')
            rea_files_with_waveforms_and_exist_waveform = rea_files_with_waveforms_and_exist_waveform +1
            event= nd.read_nordic(dir_rea + os.sep + f)[0]    
            with open(dir_rea + os.sep + f, "r") as archivo_rea:
                for linea in archivo_rea: 
                    if "RCC"  in linea:  #----- Busco la linea para extraer azimuth y distancia
                    
                        if "HZ" in linea or "BZ" in linea: 
                            contain_P= True                            
                            print(linea)
                            print('Linea 76:79 :',linea[76:79])
                            if linea[76:79] != "   " and linea[70:75] != "     ":
                                contain_Azimuth_Dist= True
                                back_azimuth = int(linea[76:79])
                                azimuth = calculate_azimut(back_azimuth)                               
                                print('Backazimuth',back_azimuth)
                                print('Azimuth',azimuth)
                                distance = float(linea[70:75])
                                print('Distance',distance)
        
            # ------------Selecciono el tiempo de P solo de RCC si existe P---------
            if contain_P == True and contain_Azimuth_Dist == True:
                cuantity_containing_P = cuantity_containing_P +1
                print('Contiene P')
                #print(event.picks)
                for pick in event.picks:                    
                    if pick.waveform_id.station_code == "RCC" :
                        if pick.phase_hint == "S" and (pick.waveform_id.channel_code == "HE" or pick.waveform_id.channel_code == "BE" or pick.waveform_id.channel_code == "HN" or pick.waveform_id.channel_code == "BN"):
                            s_time_arrival= pick.time
                            print ('Tiempo de S ',s_time_arrival)
                            
                for pick in event.picks:                    
                    if pick.waveform_id.station_code == "RCC" :
                        if pick.phase_hint == "P" and pick.waveform_id.channel_code == "HZ" or pick.waveform_id.channel_code == "BZ" :
                            p_time_arrival= pick.time                            
                            
                for pick in event.picks:                    
                    if pick.waveform_id.station_code == "RCC" :                        
                        if pick.phase_hint == "P" and pick.waveform_id.channel_code == "HZ" or pick.waveform_id.channel_code == "BZ" :
                            p_time_arrival= pick.time
                            print ('Tiempo de P',p_time_arrival)
                            start_time=p_time_arrival - time_before
                            print ('Inicio ventana de tiempo',start_time)
                            end_time= p_time_arrival + time_after
                            print ('Final ventana de tiempo',end_time)
                            # ------------- Selecciono solo la traza que contenga RCC
                            for waveForm in waveForms: 
                                print('Analizando waveform')
                                st= read(dir_wav + os.sep + waveForm) 
                                
                                tmp_rcc =  st.select(station="RCC")
                                print('el select de RCC',tmp_rcc)
                                print('cantidad de trazas:' , len(tmp_rcc))                               
                                if  len(tmp_rcc) != 0:
                                    sampling_rate = tmp_rcc.traces[0].stats.sampling_rate
                                    print('Sampling Rate',sampling_rate)
                                    if sampling_rate == sampling_rate_defined :  
                                        sampling_rate_100 = sampling_rate_100 +1
                                        print('La señal completa')
                                        #st.plot()
                                        print('Solo RCC counts')
                                        #tmp_rcc.plot()
                                        if len(tmp_rcc) > 3 :
                                            rcc = tmp_rcc.merge()
                                            print('La señal sin overlap')
                                            #rcc.plot()
                                        else:
                                            rcc = tmp_rcc
                                        
                                        
                                        #print ('rcc[0].channel', rcc[0].stats.channel)
                                        #print ('rcc[1].channel', rcc[1].stats.channel)
                                        #print ('rcc[2].channel', rcc[2].stats.channel)
                                        #print ('rcc[0].location', rcc[0].stats.location)
                                        #print ('rcc[1].channel', rcc[1].stats.location)
                                        #print ('rcc[2].channel', rcc[2].stats.location)
                                        
                                        if rcc[1].stats.channel == 'BHE' and rcc[2].stats.channel == 'BHN' :
                                            e_w = rcc[1]
                                            n_s = rcc[2]
                                            rcc.pop(2)
                                            rcc.pop(1)
                                            rcc.insert(1,n_s)
                                            rcc.insert(2,e_w)                                   
                                        
                                        rcc[0].stats.station = 'RCC'
                                        rcc[0].stats.channel = 'HHZ'
                                        rcc[0].stats.location = '00'
                                        rcc[0].stats.network = 'CW'
                                        rcc[1].stats.station = 'RCC'
                                        rcc[1].stats.channel = 'HHN'
                                        rcc[1].stats.location = '00'
                                        rcc[1].stats.network = 'CW'
                                        rcc[2].stats.station = 'RCC'
                                        rcc[2].stats.channel = 'HHE'
                                        rcc[2].stats.location = '00'
                                        rcc[2].stats.network = 'CW' 
                                        print('La señal corregida CW RCC 00 ')
                                        #rcc.plot()  

                                        pre_filt = [0.001, 0.005, 45, 50]
                                        rcc[0].remove_response(inventory=inv_Z,pre_filt=pre_filt, output="VEL", water_level=60)
                                        rcc[1].remove_response(inventory=inv_N, pre_filt=pre_filt, output="VEL", water_level=60)
                                        rcc[2].remove_response(inventory=inv_E, pre_filt=pre_filt, output="VEL", water_level=60)
                                        print('La señal en velocidad')
                                        #rcc.plot()
                                        rcc_filtered = rcc.copy()
                                        rcc_filtered.taper(max_percentage = 0.05,type='hann', max_length=None, side='left')
                                        print('La señal tapered')
                                        #rcc_filtered.plot()
                                        rcc_filtered.filter("bandpass",freqmin= freq_min_filter,freqmax= freq_max_filter)
                                        rcc_filtered.detrend(type='demean') 
                                        print('La señal filtrada y detrend')
                                        #rcc_filtered.plot()
                                        #Calculamos la relacion señal ruido solo si la phase S esta marcada en el sismograma
                                        if s_time_arrival is not None:
                                            cuantity_containing_S = cuantity_containing_S +1                                          
                                            print ('Entro al calculo de SNR')
                                            noise_window_start_time = p_time_arrival - 1                                            
                                            noise_window_end_time = p_time_arrival
                                            signal_window_start_time = s_time_arrival 
                                            signal_window_end_time = s_time_arrival + 1 

                                            
                                            
                                            tmp_z_noise_window = rcc_filtered.traces[0].slice(noise_window_start_time, noise_window_end_time)
                                            tmp_n_noise_window = rcc_filtered.traces[1].slice(noise_window_start_time, noise_window_end_time)
                                            tmp_e_noise_window = rcc_filtered.traces[2].slice(noise_window_start_time, noise_window_end_time)
                                            
                                            tmp_z_signal_window = rcc_filtered.traces[0].slice(signal_window_start_time, signal_window_end_time)
                                            tmp_n_signal_window = rcc_filtered.traces[1].slice(signal_window_start_time, signal_window_end_time)
                                            tmp_e_signal_window = rcc_filtered.traces[2].slice(signal_window_start_time, signal_window_end_time)
                                            
                                            
                                            print('tamaño de noise window :', len(tmp_z_noise_window))
                                            
                                            
                                            if len(tmp_z_signal_window) == 101 and len(tmp_n_signal_window)== 101   and len(tmp_e_signal_window)== 101 and len(tmp_z_noise_window)==101 and len(tmp_n_noise_window)== 101 and len(tmp_e_noise_window)== 101:
                                                snr_s_windows_ok = snr_s_windows_ok + 1
                                                snr_Z_S = snr(tmp_z_signal_window, tmp_z_noise_window)                                         
                                                snr_N_S = snr(tmp_n_signal_window, tmp_n_noise_window)                                         
                                                snr_E_S = snr(tmp_e_signal_window, tmp_e_noise_window)
                                                
                                                snr_2_Z_S = snr_2(tmp_z_signal_window, tmp_z_noise_window)                                         
                                                snr_2_N_S = snr_2(tmp_n_signal_window, tmp_n_noise_window)                                         
                                                snr_2_E_S = snr_2(tmp_e_signal_window, tmp_e_noise_window)
                                                print ('SNR: ',snr_Z_S, snr_N_S, snr_E_S)
                                                print ('SNR_2: ',snr_2_Z_S, snr_2_N_S, snr_2_E_S)
                                        # Calculo SNR respecto a la onda P
                                        print ('Entro al calculo de SNR para las P')
                                        noise_window_start_time = p_time_arrival - 1                                            
                                        noise_window_end_time = p_time_arrival
                                        signal_window_start_time = p_time_arrival 
                                        signal_window_end_time = p_time_arrival + 1 

                                        
                                            
                                        
                                            
                                        tmp_z_noise_window = rcc_filtered.traces[0].slice(noise_window_start_time, noise_window_end_time)
                                        tmp_n_noise_window = rcc_filtered.traces[1].slice(noise_window_start_time, noise_window_end_time)
                                        tmp_e_noise_window = rcc_filtered.traces[2].slice(noise_window_start_time, noise_window_end_time)
                                            
                                        tmp_z_signal_window = rcc_filtered.traces[0].slice(signal_window_start_time, signal_window_end_time)
                                        tmp_n_signal_window = rcc_filtered.traces[1].slice(signal_window_start_time, signal_window_end_time)
                                        tmp_e_signal_window = rcc_filtered.traces[2].slice(signal_window_start_time, signal_window_end_time)
                                            
                                            
                                        print('tamaño de noise window :', len(tmp_z_noise_window))
                                            
                                            
                                        if len(tmp_z_signal_window) == 101 and len(tmp_n_signal_window)== 101 and len(tmp_e_signal_window)== 101 and len(tmp_z_noise_window)==101 and len(tmp_n_noise_window)== 101 and len(tmp_e_noise_window)== 101:
                                            snr_p_windows_ok = snr_p_windows_ok + 1    
                                            snr_Z_P = snr(tmp_z_signal_window, tmp_z_noise_window)                                         
                                            snr_N_P = snr(tmp_n_signal_window, tmp_n_noise_window)                                         
                                            snr_E_P = snr(tmp_e_signal_window, tmp_e_noise_window)
                                            
                                            snr_2_Z_P = snr_2(tmp_z_signal_window, tmp_z_noise_window)                                         
                                            snr_2_N_P= snr_2(tmp_n_signal_window, tmp_n_noise_window)                                         
                                            snr_2_E_P = snr_2(tmp_e_signal_window, tmp_e_noise_window)
                                            print ('SNR de P ',snr_Z_P, snr_N_P, snr_E_P)
                                            print ('SNR_2: ',snr_2_Z_P, snr_2_N_P, snr_2_E_P)
                                        
                                                        
                                        rcc_sliced= rcc_filtered.slice(start_time, end_time)
                                        print('La señal velocidad filtrada y sliced')
                                        #rcc_sliced.plot(type='relative', tick_format='%H:%M:%S', starttime=rcc_sliced[0].stats.starttime, endtime=rcc_sliced[0].stats.endtime, time_down=True)                       
                        
                                        #------extraigo las magnitudes del Evento ML, Mc,MW
                                        if (len(event.magnitudes)) == 3 :
                                            ML = event.magnitudes[0].mag
                                            Mc = event.magnitudes[1].mag
                                            MW = event.magnitudes[2].mag
                                        elif (len(event.magnitudes)) == 2 :
                                            ML = event.magnitudes[0].mag
                                            Mc = event.magnitudes[1].mag
                                            MW = None
                                        elif (len(event.magnitudes)) == 1 :
                                            ML = event.magnitudes[0].mag
                                            Mc = None
                                            MW = None
                                        print('ML',ML)
                                        print('Mc',Mc)
                                        print('MW',MW)
                                        new_row =  {'Origin_Time':event.origins[0].time,
                                                    'Latitude':event.origins[0].latitude,
                                                    'Longitude':event.origins[0].longitude,
                                                    'Depth':event.origins[0].depth/1000,
                                                    'RMS':event.origins[0].quality.standard_error,
                                                    'ML':ML,
                                                    'Mc':Mc,
                                                    'MW':MW,
                                                    'P_time': p_time_arrival,
                                                    'S_time': s_time_arrival,
                                                    'Distance': distance,
                                                    'Back_Azimuth': back_azimuth,
                                                    'Azimuth': azimuth,
                                                    'WAV_file': waveForm,
                                                    'RCC_OBSPY': rcc_sliced,
                                                    'Z_data':rcc_sliced[0].data,
                                                    'N_data':rcc_sliced[1].data,
                                                    'E_data':rcc_sliced[2].data,
                                                    'SNR_Z_S': snr_Z_S,
                                                    'SNR_N_S': snr_N_S,
                                                    'SNR_E_S': snr_E_S,
                                                    'SNR_Z_P': snr_Z_P,
                                                    'SNR_N_P': snr_N_P,
                                                    'SNR_E_P': snr_E_P,
                                                    'SNR_2_Z_S': snr_2_Z_S,
                                                    'SNR_2_N_S': snr_2_N_S,
                                                    'SNR_2_E_S': snr_2_E_S,
                                                    'SNR_2_Z_P': snr_2_Z_P,
                                                    'SNR_2_N_P': snr_2_N_P,
                                                    'SNR_2_E_P': snr_2_E_P
                                                    }
                                        df = pd.concat([df, pd.DataFrame([new_row])]).reset_index(drop=True)
                                        added_to_dataframe = added_to_dataframe + 1
                                        print(df)
print("================================ Salvando los datos ====================")
print('Rea files analyzed: ', rea_files_cuantity)
print('Rea files  with waveForms: ', rea_files_with_waveforms)
print('Rea files  with waveForms and Exist the waveform: ', rea_files_with_waveforms_and_exist_waveform)
print('Rea files  with P: ', cuantity_containing_P)
print('Rea files  with S: ', cuantity_containing_S)
print('Rea files  with sampling_rate_100: ', sampling_rate_100)
print('Rea files  with snr_s_windows_ok: ', snr_s_windows_ok)
print('Rea files  with snr_p_windows_ok: ', snr_p_windows_ok)
print('Event added to dataframe: ', added_to_dataframe)

df.to_csv(dir_output + os.sep +'rcc_only_earthquake_database_20160117_replicas_1mes_v5_1.csv')
df.to_pickle(dir_output + os.sep +'rcc_only_earthquake_database_20160117_replicas_1mes_v5_1.pkl')
print("Finalizado")


