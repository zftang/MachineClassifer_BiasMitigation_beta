#!/usr/bin/env python
# coding: utf-8

import numpy as np      
import pandas as pd        
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from itertools import combinations
import copy
import math
import os

from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from params import *



def cal_min_max_normalization(df_input): 
    min_val = df_input.min()
    max_val = df_input.max()
    val_arr = df_input.values
    
    if max_val != min_val:
        df_output = pd.Series(index=df_input.index, data=(val_arr - min_val) / (max_val - min_val))   
        return df_output
    else:
        return df_input

def cal_S_numerical_and_categorical(X_input_full, y_input, numerical_attribute, categorical_attribute):
    #--------------------
    numerical_attribute_tmp = []
    for i_col in X_input_full.columns:
        if i_col in numerical_attribute:
            numerical_attribute_tmp.append(i_col)
    categorical_attribute_tmp = []
    for i_col in X_input_full.columns:
        if i_col in categorical_attribute:
            categorical_attribute_tmp.append(i_col)
    
    #---
    #numerical_attribute
    X_input = X_input_full[numerical_attribute_tmp].values
    
    #---
    X_pos = X_input[y_input==1]
    X_neg = X_input[y_input==0]
    
    #X_pos_mean = np.mean(X_pos, axis=0)
    #X_neg_mean = np.mean(X_neg, axis=0)
    X_pos_mean = np.nanmean(X_pos, axis=0)
    X_neg_mean = np.nanmean(X_neg, axis=0)    
    
    df_S_numerical = pd.Series(index=numerical_attribute_tmp, data=np.abs(X_pos_mean - X_neg_mean ))
    
    #---
    #categorical_attribute 
    Si_arr = []
    for i_categorical in categorical_attribute_tmp:
        df_i_categorical = pd.DataFrame(X_input_full[i_categorical])
        
        df_i_categorical['label'] = y_input   
        df_freq = pd.DataFrame()
        df_p = df_i_categorical[df_i_categorical['label'] == 1].groupby(i_categorical).count()
        df_n = df_i_categorical[df_i_categorical['label'] == 0].groupby(i_categorical).count()
        
        if df_n.shape[0]>=df_p.shape[0]:
            df_freq['L'] = df_n
            df_freq['S'] = df_p
        else:
            df_freq['S'] = df_p
            df_freq['L'] = df_n
        
        df_freq.fillna(0, inplace=True)
        df_freq['L'] = df_freq['L']/df_freq['L'].sum()
        df_freq['S'] = df_freq['S']/df_freq['S'].sum()
        df_diff_each_categorical = np.abs( df_freq['L']  - df_freq['S'] )
        Si_val = df_diff_each_categorical.mean()

        Si_arr.append(Si_val)
            
    df_S_categorical = pd.Series(index=categorical_attribute_tmp, data=Si_arr)
    
    return df_S_numerical, df_S_categorical

def cal_df_S(df_data_processed_input, label_O, numerical_attribute, categorical_attribute):
    
    df_data_attribute = df_data_processed_input.drop(columns=[label_O]).copy()
    df_label_O = df_data_processed_input[label_O].copy() 
    label_O_val = df_label_O.values
    
    # calculate the S_m for numerical and categorical attributes 
    df_S_numerical_tmp, df_S_categorical_tmp =     cal_S_numerical_and_categorical(df_data_attribute, label_O_val, numerical_attribute, categorical_attribute)    
    df_S = pd.concat([df_S_numerical_tmp, df_S_categorical_tmp])
    df_S.to_frame().T
    
    return df_S


def get_upper(val_input):

    str_val =  '{:.8f}'.format(np.float32(val_input))
    if val_input<0.5:
        counter = 0
        for i in str_val.split('.')[1]:
            if i != '0':
                break
            counter+=1 
        if np.int32(i) >= 5:
            val_output = np.float64('0.' + '0'*(counter-1) + '1' )
        elif np.int32(i) >= 2:
            val_output = np.float64('0.' + '0'*(counter) + '5' )
        else:
            val_output = np.float64('0.' + '0'*(counter) + '2' )
        #print(val_input, val_output)
    elif val_input>=0.5 and val_input<1:
        val_output = 1
        
    else:    
        str_tmp = str_val.split('.')[0]
        #pass
        i = str_tmp[0]
        if np.int32(i) >= 5:
            val_output = np.float64('1'+'0'*(len(str_tmp)))
        elif np.int32(i) >= 2:
            val_output = np.float64('5'+'0'*(len(str_tmp)-1))
        else:
            val_output = np.float64('2'+'0'*(len(str_tmp)-1))    
    return val_output







ResultPath = 'Result/' + DatasetName + '/'
try:
    os.mkdir('Result/')
except:
    pass     
try:
    os.mkdir(ResultPath)
except:
    pass      









#--------------------------
# read data
df_data = pd.read_csv(DataPath, header=None)
df_data.columns = data_attribute




#--------------------------
print('label_Y is : '+label_Y)
print('label_O is : '+label_O)



#---
# encode the categorical attributes
le = LabelEncoder()
LabelEncoder_Cols = categorical_attribute

df = df_data[LabelEncoder_Cols].copy()
d = defaultdict(LabelEncoder)
encoded_series = df.apply(lambda x: d[x.name].fit_transform(x))

df_data_processed = pd.concat([df_data.drop(LabelEncoder_Cols,axis=1), encoded_series], axis=1)
df_data_processed_bk = (pd.concat([df_data_processed[numerical_attribute].astype(np.float64),  df_data_processed[categorical_attribute+[label_Y]]], axis=1)).copy() 
 



epsilon_threshold_val = -np.inf
debug_mode = False
step_val = 0
bias_mitigation_dict = {}
attribute_modified = None

epsilon_val = 0
polynomial_i = 0
polynomial_arr = np.array([ [i, 1/i] for i in range(3,2000,2)]).reshape(-1) 






if weighted_bool:
    try:
        os.mkdir(ResultPath+'/Weighted/')
    except:
        pass
    

print('-----------------')
counter = 0
step_info_arr= ['Raw'] 
no_info_arr = []


while (epsilon_val > epsilon_threshold_val) and (counter<1000):
    counter+=1
    
    
    print('Current Step=', step_val,  'and Iter=',  counter)
    
    if debug_mode:
        
        if attribute_modified in categorical_attribute:
            
            df_1 = df_data_processed[[attribute_modified, label_O]][df_data_processed[label_O]==1].groupby(attribute_modified).count()
            df_2 = df_data_processed[[attribute_modified, label_O]][df_data_processed[label_O]==0].groupby(attribute_modified).count()
            df_c = (df_1/(df_1.sum()) - df_2/(df_2.sum())).fillna(0).sort_values(by=label_O)            
            
            print('Bias mitigation for categorical_attribute: ',attribute_modified,  ' re-bin:',  {df_c.index[-1]:df_c.index[0]})
            
            
            str_tmp = ' re-bin: ' + '{' + str(d[attribute_modified].inverse_transform([df_c.index[-1]])[0])              + ',' +  str(d[attribute_modified].inverse_transform([df_c.index[0]])[0]) + '}'
            print(str_tmp)
            step_info_arr.append(attribute_modified + ' Iter' + str(counter-1)+ str_tmp)
            
            if attribute_modified in bias_mitigation_dict.keys():
                bias_mitigation_dict[attribute_modified].update({df_c.index[-1]:df_c.index[0]})
            else:
                bias_mitigation_dict[attribute_modified] = {df_c.index[-1]:df_c.index[0]}
                
        
        
        elif attribute_modified in numerical_attribute: 
            
            polynomial_val = polynomial_arr[polynomial_i]

            df_tmp = df_data_processed_bk[attribute_modified].copy()
            df_tmp = ((df_tmp.abs())**(polynomial_val)) * (df_tmp.apply(np.sign))

            if df_tmp.abs().max() > np.finfo(np.float32).max:
                print('Bias mitigation for numerical_attribute: ',attribute_modified,  '; polynomial:',  polynomial_arr[polynomial_i])
                bias_mitigation_dict[attribute_modified] = 'inf'
                
                str_tmp = ' polynomial: ' + 'inf'
                print(str_tmp)            
                step_info_arr.append(attribute_modified + ' Step' + str(counter-1)+ str_tmp)                    
            else:
                print('Bias mitigation for numerical_attribute: ',attribute_modified,  '; polynomial:',  polynomial_arr[polynomial_i])
                bias_mitigation_dict[attribute_modified] = polynomial_arr[polynomial_i]
            
                str_tmp = ' polynomial: ' + str(polynomial_arr[polynomial_i])
                print(str_tmp)            
                step_info_arr.append(attribute_modified + ' Step' + str(counter-1)+ str_tmp)            
            
            
        else:
            pass
    
    else:
        pass

    
    df_data_processed = df_data_processed_bk.copy()
    df_data_processed.drop(columns=[label_Y], inplace=True)


    bias_mitigation_keys_arr = list(bias_mitigation_dict.keys())
    for i_attribute in bias_mitigation_keys_arr:

        dict_attribute = bias_mitigation_dict[i_attribute]


        if i_attribute in numerical_attribute:
            
            if dict_attribute == 'inf':
                y_name = i_attribute
                df_data_processed[y_name] = 1 #y_data  
                print(i_attribute, ' polynomial transformation failed, drop it') 
                
                no_info_arr.append(i_attribute)
                
            else:
                y_name = i_attribute
                #y_data = df_data_processed[y_name]**(dict_attribute)
                y_data = ((df_data_processed[y_name].abs())**(dict_attribute)) * (df_data_processed[y_name].apply(np.sign))
                df_data_processed[y_name] = y_data  

            
            
            
        # categorical attribute: re-binning transformation
        elif i_attribute in categorical_attribute:
            y_name = i_attribute
            df_tmp = df_data_processed[y_name].copy() 
            keys_arr = list(dict_attribute.keys())
            for i_key in keys_arr:
                df_tmp[df_tmp==i_key] = dict_attribute[i_key]  
            df_data_processed[y_name] = df_tmp
            
            if i_key == dict_attribute[i_key]  :
                no_info_arr.append(i_attribute)
                print(i_attribute, ' categorical transformation failed, drop it') 
            
            
        else:
            raise Exception('Error attribute', i_attribute)

            
            
    #-----------------------------------------------------------------------------------
    # Step1: Data Normalization
    comb_label_arr = list(combinations(list(set(df_data_processed[label_O].values)), 2))
    df_S_full = pd.DataFrame()
    for comb_label in comb_label_arr:

        # For multi-nary label_O, select a pair of data each time
        p_label, n_label = comb_label
        df_data_processed_p = df_data_processed[df_data_processed[label_O]==p_label].copy()
        df_data_processed_n = df_data_processed[df_data_processed[label_O]==n_label].copy()
        df_data_processed_p[label_O] = 1
        df_data_processed_n[label_O] = 0    
        df_data_processed_tmp = pd.concat([df_data_processed_p, df_data_processed_n])

        # calculate the min-max normalization
        for i_attribute in numerical_attribute:
            df_tmp = cal_min_max_normalization(df_data_processed_tmp[i_attribute])
            df_data_processed_tmp[i_attribute] = df_tmp

        ## compute attributes’ contribution
        df_S_tmp = cal_df_S(df_data_processed_tmp, label_O, numerical_attribute, categorical_attribute) 
        df_S_full[comb_label] = df_S_tmp    

    #-----------------------------------------------------------------------------------
    #Step 2: Distance Matrix Construction
    ## Compute sub-distance and overall distance
    attribute_num = df_data_processed.shape[1] - 1

    index_arr = list(df_S_full.index) + ['origin']
    distance_matrix = pd.DataFrame(index=index_arr, columns=index_arr, data=np.nan)
    for x_a in range(attribute_num+1):
        distance_matrix.iloc[x_a, x_a] = 0

    # x_empty represents the origin 
    x_empty = attribute_num
    wmax_arr = []        

    def cal_mat_val(x_a, x_b):        

        if (x_a == x_empty) and (x_b == x_empty):
            pass
        else:
            #print('----------')
            #print('calculating:', index_arr[x_a], ':', x_a, ';', index_arr[x_b],  x_b)
            sub_set_arr = [x for x in range(attribute_num)]
            if x_a == x_empty :
                pass
            else:
                sub_set_arr.remove(x_a)
            if x_b == x_empty :
                pass
            else:
                sub_set_arr.remove(x_b) 

            #-------------------------------
            N_total = len(sub_set_arr)
            if h_order_val==-1:
                h_order = N_total 
            elif h_order_val > N_total:
                raise Exception('Error h_order_val', h_order_val)
            else:
                h_order = h_order_val
            Terminal_H = N_total - h_order  


            # dist_val_arr is used to store the value of each sub-distance
            dist_val_arr = []
            for sub_set_size in range(N_total, Terminal_H-1, -1):

                comb_arr = list(combinations([x for x in range(N_total)], sub_set_size))
                #print('--------sub_distance--------')
                #print(sub_set_size, len(comb_arr))

                for i_comb in range(len(comb_arr)):
                    sub_set_arr_tmp = copy.deepcopy([sub_set_arr[i] for i in comb_arr[i_comb]]) 


                    full_set_arr = sub_set_arr_tmp + [x_a] + [x_b]
                    if x_empty in full_set_arr:
                        full_set_arr.remove(x_empty)    

                    #-----------------------------
                    # remove x_a    
                    full_set_arr_tmp = copy.deepcopy(full_set_arr)
                    if x_a == x_empty:
                        pass
                    else:
                        full_set_arr_tmp.remove(x_a)

                    wmax_a_arr = []
                    for comb_label in comb_label_arr:
                        df_S_input = df_S_full[comb_label]
                        df_tmp = df_S_input.iloc[full_set_arr_tmp].copy()
                        if df_tmp.shape[0]>0:
                            wmax_a = np.sqrt( (df_tmp**2).sum() / df_tmp.shape[0])
                        else:
                            wmax_a = 0

                        wmax_a_arr.append(wmax_a)
                    wmax_a = np.max(wmax_a_arr)


                    #-----------------------------
                    # remove x_b
                    full_set_arr_tmp = copy.deepcopy(full_set_arr)
                    if x_b == x_empty:
                        pass
                    else:
                        full_set_arr_tmp.remove(x_b)

                    wmax_b_arr = []
                    for comb_label in comb_label_arr:
                        df_S_input = df_S_full[comb_label]
                        df_tmp = df_S_input.iloc[full_set_arr_tmp].copy()
                        if df_tmp.shape[0]>0:
                            wmax_b = np.sqrt( (df_tmp**2).sum() / df_tmp.shape[0])
                        else:
                            wmax_b = 0

                        wmax_b_arr.append(wmax_b)
                    wmax_b = np.max(wmax_b_arr)

                    #-----------------------------
                    
                    if weighted_bool == True:
                        weight_val = 1/ (    comb(N_total, sub_set_size) * comb(attribute_num, sub_set_size) )
                        dist_val = np.abs(wmax_a - wmax_b)
                        dist_val_arr.append(dist_val*weight_val)                  
                    else:
                        dist_val = np.abs(wmax_a - wmax_b)
                        dist_val_arr.append(dist_val)

            # the full distance matrix is the un-weighted sum (i.e. mean) of all sub-distance
            #distance_matrix.iloc[x_a, x_b] = np.mean(dist_val_arr)
            #distance_matrix.iloc[x_b, x_a] = distance_matrix.iloc[x_a, x_b]
            if weighted_bool == True:
                distance_matrix_val = np.sum(dist_val_arr)
            else:
                distance_matrix_val = np.mean(dist_val_arr)

        return (x_a, x_b, distance_matrix_val)

    x_ab_arr = []
    ii = 0
    for x_a in range(attribute_num+1):
        for x_b in range(x_a+1, attribute_num+1):
            x_ab_arr.append((x_a, x_b))
            ii+=1
    #print(ii)

    result_tmp = Parallel(n_jobs=num_jobs)(
        delayed(cal_mat_val)(x_a, x_b)
        for x_a, x_b in x_ab_arr)

    for x_a, x_b, distance_matrix_val in result_tmp:
        distance_matrix.iloc[x_a, x_b] = distance_matrix_val
        distance_matrix.iloc[x_b, x_a] = distance_matrix_val

    #-----------------------------------------------------------------------------------
    #Step 3:  Bias Concentration Determination
    ## MDS
    random_state_val = 0
    stress = []
    # Max value for n_components
    max_range = 9
    for dim in range(1, max_range):
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state_val, n_init=4, max_iter=10000, eps=1e-10)
        mds.fit(distance_matrix)
        stress.append(mds.stress_) 


    # elbow plot
    fontsize_val = 13
    fig = plt.figure()
    ax = pd.Series(index=range(1, max_range), data=stress).plot(style='o-', grid=True, fontsize=fontsize_val)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.set_xlabel('dim (MDS)', fontsize=fontsize_val)
    ax.set_ylabel('Stress', fontsize=fontsize_val)

    fig.savefig(ResultPath+'MDS_elbow_step'+str(step_val)+'.png',bbox_inches = 'tight')


    ## projected attribute space (2D)
    # here the MDS dimensition is fixed at 2; for higher dimension, process the 'distance_matrix_xxx.csv' in Result path
    fig = plt.figure()

    random_state_val = 1
    dim = 2
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state_val, n_init=4, max_iter=10000, eps=1e-10)

    # Apply MDS
    # pts = mds.fit(distance_matrix) 
    pts = mds.fit_transform(distance_matrix)
    pts  = pts - pts[-1] 

    ax = pd.Series(index=pts[:, 0], data=pts[:, 1]).plot(style='o', grid=True, fontsize=fontsize_val)
    for i in range(len(distance_matrix)):
        ax.text(pts[i, 0]  , pts[i, 1] , distance_matrix.columns[i], fontsize=12)
    i = attribute_num
    pd.Series(index=[pts[i, 0]], data=[pts[i, 1]]).plot(ax=ax,style='rx')
    ax.text(pts[i, 0]  , pts[i, 1] , distance_matrix.columns[i], fontsize=12, color='red')
    ax.legend(['MDS Coordinates (dim=2)'], fontsize=fontsize_val)
    ax.grid()
    ax.set_xlabel('axis 1', fontsize=fontsize_val)
    ax.set_ylabel('axis 2', fontsize=fontsize_val)


    fig.savefig(ResultPath+'MDS_2D_step'+str(step_val)+'.png',bbox_inches = 'tight')

    #----------------------
    # calculate the distance to origin
    bias_concentration_matrix = pd.DataFrame(data=pts, index=distance_matrix.index).T
    df_dist2origin = (bias_concentration_matrix**2).sum().apply(np.sqrt)

    
    
    
    
    
    
    # # # ! ! !epsilon_threshold_val
    if step_val == 0:
        X = df_dist2origin.values.reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        kmeans.labels_
        

        K0_mean_val = df_dist2origin[~np.bool8(kmeans.labels_)].mean()
        # rm the origin
        #K0_mean_val = np.mean(df_dist2origin[~np.bool8(kmeans.labels_)].values[:-1])
        
        dist_max = df_dist2origin.max()
        dist_rm_max_mean = np.mean(df_dist2origin.sort_values().values[:-1])
        (dist_max/K0_mean_val)>1.5
        
        epsilon_threshold_val = get_upper(K0_mean_val)
    
    #epsilon_threshold_val = 0.01

    if epsilon_threshold_val_fix_bool:
        epsilon_threshold_val = epsilon_threshold_val_fix
    

    fig = plt.figure()
    ax = df_dist2origin.plot(style='o-')
    pd.Series(index=df_dist2origin.index, data=epsilon_threshold_val).plot(ax=ax, style='--')


    tickslabel_arr = list(df_dist2origin.index)
    tickslabel_arr = [x.replace('-', ' ') for x in tickslabel_arr]
    ax.set_xticks(range(len(tickslabel_arr)))
    ax.set_xticklabels(tickslabel_arr, rotation=90)
    ax.set_ylabel('distance to origin',fontsize=fontsize_val)
    ax.set_xticks(range(len(tickslabel_arr)))
    ax.set_xticklabels(tickslabel_arr, rotation=90) 
    ax.grid(axis="y")

    
    fig.savefig(ResultPath+'distance2origin_step'+str(step_val)+'.png', bbox_inches='tight')


    distance_matrix.to_csv(ResultPath+'distance_matrix_step_'+str(step_val)+'.csv')
    bias_concentration_matrix.to_csv(ResultPath+'bias_concentration_matrix_step_'+str(step_val)+'.csv')

    
    #---------------------------------
    epsilon_val = df_dist2origin.max() 
    
    if debug_mode != True:
        attribute_modified = df_dist2origin.sort_values().index[-1]
        step_val+=1
        debug_mode = True
        polynomial_i = 0
    
    
    else: 
        
        if attribute_modified in categorical_attribute:
            print(df_dist2origin.loc[attribute_modified] , epsilon_threshold_val)
            
            if (attribute_modified in no_info_arr) or (df_dist2origin.loc[attribute_modified] < epsilon_threshold_val):
                attribute_modified = df_dist2origin[~df_dist2origin.index.isin(no_info_arr)].sort_values().index[-1]
                step_val+=1
                debug_mode = True                
        
        
        elif attribute_modified in numerical_attribute:
            print(df_dist2origin.loc[attribute_modified] , epsilon_threshold_val)
            if (attribute_modified in no_info_arr) or (df_dist2origin.loc[attribute_modified] < epsilon_threshold_val):
                attribute_modified = df_dist2origin[~df_dist2origin.index.isin(no_info_arr)].sort_values().index[-1]
                step_val+=1
                debug_mode = True
                polynomial_i = 0     
            else:
                polynomial_i+=1
                     
        else:
            pass        
        

    
    
    #print('###---')
    print('maximum distance to origin: ',  epsilon_val, '  threshold: ', epsilon_threshold_val)
    if epsilon_val < epsilon_threshold_val:
        print('--------------------------')
        print('the maximum distance to origin is smaller than the threshold epsilon value')
        print('Bias Mitigation Progress: Finish')
        
        
        
    
    plt.cla() 
    plt.clf() 
    plt.close("all")
    
    
    #print(debug_mode)
    print('Attribute Dropped: ',  set(no_info_arr))
    print('--------------------------')


step_val_full = step_val    



bias_mitigation_dict




fig, ax = plt.subplots(nrows=1,
                       ncols=1,
                       gridspec_kw={'height_ratios': [1]})


for step_val in range(step_val_full-1):
    #print(step_val)
    df_tmp = pd.read_csv(ResultPath+'bias_concentration_matrix_step_'+str(step_val)+'.csv', index_col=0)
    df_dist2origin = (df_tmp**2).sum().apply(np.sqrt)
    df_dist2origin.plot(style='o-', ax=ax, alpha=0.3)
    
step_val = step_val_full-1
#print(step_val)
df_tmp = pd.read_csv(ResultPath+'bias_concentration_matrix_step_'+str(step_val)+'.csv', index_col=0)
df_dist2origin = (df_tmp**2).sum().apply(np.sqrt)
df_dist2origin.plot(style='o-', ax=ax, alpha=0.3)    



#print('max distance=', epsilon_threshold_val,   np.max(df_dist2origin) )    
max_distance = np.max(df_dist2origin)
df_dist2origin.to_csv(ResultPath+'bias_concentration_matrix_epsilon_'+str(epsilon_threshold_val_fix)+'.csv')
    

    
    
    
pd.Series(index=df_dist2origin.index, data=epsilon_threshold_val).plot(ax=ax, style='--') 


df_tmp = pd.read_csv(ResultPath+'bias_concentration_matrix_step_'+str(0)+'.csv', index_col=0)
df_dist2origin = (df_tmp**2).sum().apply(np.sqrt)
df_dist2origin[~np.bool8(kmeans.labels_)] = np.nan
df_dist2origin.plot(ax=ax, style='o', ms=0.1)

tickslabel_arr = list(df_dist2origin.index)
tickslabel_arr = [x.replace('-', ' ') for x in tickslabel_arr]
ax.set_xticks(range(len(tickslabel_arr)))
ax.set_xticklabels(tickslabel_arr, rotation=90)
ax.set_ylabel('distance to origin',fontsize=fontsize_val)
ax.set_xticks(range(len(tickslabel_arr)))
ax.set_xticklabels(tickslabel_arr, rotation=90) 
ax.grid(axis="y")

ax.legend(['step '+str(i) for i in range(step_val_full)])
ax.set_title(DatasetName)


if epsilon_threshold_val_fix_bool:
    fig.savefig(ResultPath+'distance2origin_'+str(epsilon_threshold_val_fix)+'.png', bbox_inches='tight')
else:
    fig.savefig(ResultPath+'distance2origin.png', bbox_inches='tight')



step_info_arr




df_data_info = pd.DataFrame(index=[DatasetName])
df_data_info['SampleNum'] = df_data_processed_bk.shape[0]
df_data_info['$\epsilon$' ] = epsilon_threshold_val
df_data_info['BiasMitigation'] = str(bias_mitigation_dict)
df_data_info['AttributeNum'] = len(categorical_attribute ) + len(numerical_attribute)
df_data_info['ProtecedAttribute'] = label_O
df_data_info['MaxDistance'] = max_distance
df_data_info['StepInfo'] = [step_info_arr]

if epsilon_threshold_val_fix_bool:
    df_data_info.to_csv(ResultPath+'result_info_'+str(epsilon_threshold_val_fix)+'.csv')
    df_data_info
else:
    df_data_info.to_csv(ResultPath+'result_info.csv')
    df_data_info



    
    

    
    
    
    
    
    

    
    
print('Generate the data for model training')

if DatasetName == 'UCIAdult':
    ProcessedDataPath = 'Data/UCIAdult/Processed/'
    try:
        os.mkdir(ProcessedDataPath)
    except:
        pass

    for process_type in ['train','test']:
        print(process_type)
        #--------------------------
        # read data
        if process_type == 'train':
            df_data = pd.read_csv('Data/UCIAdult/adult.data.csv', header=None)
        elif process_type == 'test':
            df_data = pd.read_csv('Data/UCIAdult/adult.test.csv', header=None)   



        df_data.columns = data_attribute

        #---
        # encode the categorical attributes
        le = LabelEncoder()
        LabelEncoder_Cols = categorical_attribute

        df = df_data[LabelEncoder_Cols].copy()
        d = defaultdict(LabelEncoder)
        encoded_series = df.apply(lambda x: d[x.name].fit_transform(x))

        df_data_processed = pd.concat([df_data.drop(LabelEncoder_Cols,axis=1), encoded_series], axis=1)
        df_data_processed_bk = df_data_processed.copy() 


        if epsilon_threshold_val_fix_bool:
            df_data_info = pd.read_csv(ResultPath+'result_info_'+str(epsilon_threshold_val_fix)+'.csv', index_col=0)
        else:
            df_data_info = pd.read_csv(ResultPath+'result_info.csv', index_col=0)
            df_data_info       


        #df_data_info = pd.read_csv(ResultPath+'result_info.csv', index_col=0)
        #df_data_info

        exec_str_tmp = df_data_info['BiasMitigation'].values[0]
        exec_str = 'bias_mitigation_dict='+exec_str_tmp
        exec(exec_str)
        bias_mitigation_dict      









        
        #---------------------------------------
        df_data_processed = df_data_processed_bk.copy() 

        bias_mitigation_keys_arr = list(bias_mitigation_dict.keys())
        for i_attribute in bias_mitigation_keys_arr:

            dict_attribute = bias_mitigation_dict[i_attribute]

            # numerical attribute: polynomial transformation  
            if i_attribute in numerical_attribute:

                if dict_attribute == 'inf':

                    y_name = i_attribute 
                    df_data_processed[y_name] = 1    

                else:

                    y_name = i_attribute
                    #y_data = df_data_processed[y_name]**(dict_attribute)
                    y_data = ((df_data_processed[y_name].abs())**(dict_attribute)) * (df_data_processed[y_name].apply(np.sign))
                    df_data_processed[y_name] = y_data



            # categorical attribute: re-binning transformation
            elif i_attribute in categorical_attribute:
                y_name = i_attribute
                df_tmp = df_data_processed[y_name].copy() 
                keys_arr = list(dict_attribute.keys())
                for i_key in keys_arr:
                    df_tmp[df_tmp==i_key] = dict_attribute[i_key]  
                df_data_processed[y_name] = df_tmp
            else:
                raise Exception('Error attribute', i_attribute)


          
        #---------------------------------------
        file_name = 'data_'+ process_type + '_Mitigation.csv'
        df_tmp = df_data_processed.astype(np.float64).drop(columns=[label_O])
        #print(df_tmp.shape)
        df_tmp.to_csv(ProcessedDataPath + file_name)

        file_name = 'data_'+ process_type + '_Removal.csv'
        df_tmp = df_data_processed_bk.astype(np.float64).drop(columns=[label_O])
        #print(df_tmp.shape)
        df_tmp.to_csv(ProcessedDataPath + file_name)

        file_name = 'data_'+ process_type + '_Raw.csv'
        df_tmp = df_data_processed_bk.astype(np.float64)
        #print(df_tmp.shape)
        df_tmp.to_csv(ProcessedDataPath + file_name)      
        
else:
    raise Exception('Modified the read_data part for train/test if the Dataset is not the Default UCIAdult') 
    
    
print('Finished')