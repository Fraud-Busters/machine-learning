from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score , roc_curve, auc, average_precision_score

app = Flask(__name__)

# Load the trained Random Forest model
model = ExtraTreesClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Preprocess the data
    df_fraud = pd.DataFrame(data)

    # Select only the relevant source
    channel = ['INCOMPLETE_CS_REPORT_VICTIM', 'INCOMPLETE_CS_REPORT_SCAMMER', 'CS_REPORT_SCAMMER', 'CS_REPORT_VICTIM']
    df_fraud = df_fraud[df_fraud['source'].apply(lambda x : x in channel)]

    # Split the data into train and test and validation
    # Set split ratio
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.7, 0.1, 0.2
    VAL_SIZE = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

    # Split dataset into train and test
    df_fraud, df_fraud_val, y, y_val = train_test_split(df_fraud.drop(['is_scammer'], axis=1), df_fraud['is_scammer'],
                                                        train_size = TRAIN_SIZE, random_state = 210301)
    
    # Drop duplicate data
    df_fraud,y = handle_duplicate(df_fraud,y)

    # Fill missing values
    df_fraud, df_fraud_val = fill_missing_values(df_fraud, df_fraud_val)

    # Remove duplicate categories
    df_fraud, df_fraud_val = remove_duplicate_categories(df_fraud, df_fraud_val)

    # Feature selection
    df_fraud, df_fraud_val = remove_redundant_features(df_fraud,df_fraud_val)

    # Add new feature
    df_fraud, df_fraud_val = add_new_feature(df_fraud,df_fraud_val)

    # Feature Encoding
    df_fraud, df_fraud_val = feature_encoding(df_fraud, df_fraud_val)
    
    # Feature scaling
    df_fraud, df_fraud_val = feature_scaling(df_fraud, df_fraud_val)
    
    # Select column to use
    used_columns_9 = []
    # job_positions OHE
    used_columns_9 += [col for col in df_fraud.columns if "job_position_" in col[:13]]
    # use logged numerical features
    used_columns_9 += [col for col in df_fraud.columns if "std_" in col[:13]]
    # EDA features
    used_columns_9 += ['gender(num)', 'has_null', 'account_lifetime', 'is_group_1', 'is_group_2', 'is_group_3']

    # use 
    print('Used columns in this model:')
    used_columns_9

    ext1 = ExtraTreesClassifier()

    # Make predictions
    predictions, precision_bias, precision_var = train_and_evaluate(((df_fraud[used_columns_9].astype(np.float64)).replace([np.inf, -np.inf], 0.0).fillna(0.0), y,
                   (df_fraud_val[used_columns_9].astype(np.float64)).replace([np.inf, -np.inf], 0.0).fillna(0.0), y_val,
                   ext1, 'ext1'))
    
    # Return the predictions as a response
    response = {'predictions': predictions.tolist(), 'precision_bias': precision_bias.tolist(), 'precision_var': precision_var.toList()}
    return jsonify(response)





# Define Functions
def handle_duplicate(df_fraud,y):
    duplicated_index = df_fraud.duplicated()
    duplicated_index = duplicated_index[duplicated_index].index
    duplicates = df_fraud[df_fraud.duplicated()]
    if not duplicates.empty:
        df_fraud.drop(labels=duplicated_index, inplace=True)
        y.drop(labels=duplicated_index, inplace=True)
    return df_fraud, y


def fill_missing_values(df_fraud, df_fraud_val):
    # Fill missing values in the DataFrame
    # impute null values in numerical features with median
    num_imputer = SimpleImputer(strategy='median')
    num_cols = df_fraud.select_dtypes(include=['float64']).columns.tolist()
    df_fraud[num_cols] = num_imputer.fit_transform(df_fraud[num_cols])

    # impute null values in categorical features with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = df_fraud.select_dtypes(include=['object']).columns.tolist()
    df_fraud[cat_cols] = cat_imputer.fit_transform(df_fraud[cat_cols])

    # Also impute for validation dataset
    df_fraud_val[num_cols] = num_imputer.transform(df_fraud_val[num_cols])
    df_fraud_val[cat_cols] = cat_imputer.transform(df_fraud_val[cat_cols])

    return df_fraud, df_fraud_val

def map_job(job):
    if job not in map:
        return 'OTHERS'
    return map[job]

def remove_duplicate_categories(df_fraud, df_fraud_val):
    map= {
    'PELAJAR / MAHASISWA': 'PELAJAR / MAHASISWA',
    'MENGURUS RUMAH TANGGA': 'MENGURUS RUMAH TANGGA',
    'BELUM / TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'KARYAWAN SWASTA': 'KARYAWAN SWASTA',
    'WIRASWASTA': 'WIRASWASTA',
    'BURUH HARIAN LEPAS': 'BURUH HARIAN LEPAS',
    'PETANI / PEKEBUN': 'PETANI / PEKEBUN',
    'PEDAGANG': 'PEDAGANG',
    'PEGAWAI NEGERI SIPIL': 'PEKERJA PEMERINTAH',
    '13': '__NUMBER__',
    'BURUH TANI / PERKEBUNAN': 'PETANI / PEKEBUN',
    'KARYAWAN HONORER': 'KARYAWAN HONORER',
    'GURU': 'GURU',
    'PERDAGANGAN': 'PEDAGANG',
    'KARYAWAN BUMN': 'PEKERJA PEMERINTAH',
    '131': '__NUMBER__',
    'SOPIR': 'SOPIR',
    'NELAYAN / PERIKANAN': 'NELAYAN / PERIKANAN',
    'PEKERJAAN LAINNYA': 'OTHERS',
    '110': '__NUMBER__',
    '16': '__NUMBER__',
    'PENSIUN': 'PENSIUN',
    'KEPOLISIAN RI': 'PEKERJA PEMERINTAH',
    'BIDAN': 'BIDAN',
    'TENTARA NASIONAL INDONESIA': 'PEKERJA PEMERINTAH',
    'PERAWAT': 'PERAWAT',
    'PERANGKAT DESA': 'PEKERJA PEMERINTAH',
    'TUKANG JAHIT': 'TUKANG JAHIT',
    'TUKANG KAYU': 'TUKANG KAYU',
    'DOKTER': 'DOKTER',
    'MEKANIK': 'MEKANIK',
    'KARYAWAN BUMD': 'PEKERJA PEMERINTAH',
    'DOSEN': 'DOSEN',
    'PEMBANTU RUMAH TANGGA': 'PEMBANTU RUMAH TANGGA',
    '156': '__NUMBER__',
    '114': '__NUMBER__',
    'TUKANG BATU': 'TUKANG BATU',
    'PELAUT': 'PELAUT',
    'PELAJAR/MAHASISWA': 'PELAJAR / MAHASISWA',
    'INDUSTRI': 'INDUSTRI',
    'WARTAWAN': 'WARTAWAN',
    'PETERNAK': 'PETERNAK',
    'SENIMAN': 'SENIMAN',
    'TUKANG LAS / PANDAI BESI': 'TUKANG LAS / PANDAI BESI',
    'PENATA RAMBUT': 'PENATA RAMBUT',
    'TRANSPORTASI': 'TRANSPORTASI',
    'KONSTRUKSI': 'KONSTRUKSI',
    'PENDETA': 'PENDETA',
    'BURUH NELAYAN / PERIKANAN': 'BURUH NELAYAN / PERIKANAN',
    'PENGACARA': 'PENGACARA',
    'PENSIUNAN': 'PENSIUNAN',
    'USTADZ / MUBALIGH': 'USTADZ / MUBALIGH',
    'TUKANG CUKUR': 'TUKANG CUKUR',
    'KONSULTAN': 'KONSULTAN',
    'BELUM TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'WIRASWASAT': 'WIRASWASTA',
    '68': '__NUMBER__',
    'BELUM/TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'APOTEKER': 'APOTEKER',
    'TUKANG LISTRIK': 'TUKANG LISTRIK',
    'ANGGOTA LEMBAGA TINGGI LAINNYA': 'PEKERJA PEMERINTAH',
    'PILOT': 'PILOT',
    'WIRASAWSTA': 'WIRASWASTA',
    'WIRSWASTA': 'WIRASWASTA',
    'WIASRWASTA': 'WIRASWASTA',
    'WIRASWATA': 'WIRASWASTA',
    'WIRAWASTA': 'WIRASWASTA',
    'MENGURUS RUMAH': 'MENGURUS RUMAH TANGGA',
    'PEKERJAAN LAINNTA': 'OTHERS',
    'PENATA RIAS': 'PENATA RIAS',
    '126': '__NUMBER__',
    '112': '__NUMBER__',
    'PENTERJEMAH': 'PENTERJEMAH',
    'NOTARIS': 'NOTARIS',
    'PEKERJA LAINNYA': 'OTHERS',
    'WIRASWSASTA': 'WIRASWASTA',
    'PERANCANG BUSANA': 'PERANCANG BUSANA',
    'PENATA BUSANA': 'PERANCANG BUSANA',
    'PEKERJAAN LAINYYA': 'OTHERS',
    'SWASTA': 'KARYAWAN SWASTA',
    'AKUNTAN': 'AKUNTAN',
    'GUBERNUR': 'PEKERJA PEMERINTAH',
    'WIARSWASTA': 'WIRASWASTA',
    'OTHERS': 'OTHERS',
    np.nan : 'NULL'
    }

    # Map job position for train dataset
    df_fraud['job_position'] = df_fraud['job_position'].apply(lambda s : map_job(s))

    # Map job position for val dataset
    df_fraud_val['job_position'] = df_fraud_val['job_position'].apply(lambda s : map_job(s))

    low_count_job = df_fraud['job_position'].value_counts() <= (100)
    low_count_job = low_count_job[low_count_job].index
    df_fraud['job_position'] = df_fraud['job_position'].apply(lambda x : 'OTHERS' if x in low_count_job else x)

    return df_fraud, df_fraud_val

def remove_redundant_features(df_fraud,df_fraud_val):
    # Define a list of redundant features to drop
    redundant_features = [
        'aqc_mean_topup_amount',
        'aqc_mean_topup_amount_7d',
        'aqc_mean_topup_amount_30d',
        'aqc_total_topup_amount_90d',
        'aqc_freq_x2x_within_90d',
        'aqc_mean_x2x_amount',
        'aqc_mean_x2x_amount_7d',
        'aqc_mean_x2x_amount_30d',
        'aqc_mean_x2x_amount_60d',
        'aqc_total_x2x_amount_7d',
        'aqc_total_x2x_amount_30d',
        'aqc_total_x2x_amount_60d',
        'aqc_total_x2x_amount_90d',
        'centrality_undirected_p2p'
    ]

    # Drop the redundant features from the training dataset
    df_fraud = df_fraud.drop(columns=redundant_features)

    # Drop the redundant features from the validation dataset
    df_fraud_val = df_fraud_val.drop(columns=redundant_features)

    # Define a list of redundant features to drop
    redundant_features = [
        'avg_topup_weight_1',
        'aqc_freq_x2x',
        'avg_x2x_weight_1'
    ]

    # Drop the redundant features from the training dataset
    df_fraud = df_fraud.drop(columns=redundant_features)

    # Drop the redundant features from the validation dataset
    df_fraud_val = df_fraud_val.drop(columns=redundant_features)
    
    return df_fraud, df_fraud_val

def assign_group(in_, out_, threshold):
  """Assign a group to a data based on the ratio of in_ / out_"""
  if in_ > out_ * threshold:
    return 1
  elif in_ * threshold < out_:
    return 2
  else:
    return 3

def add_new_feature(df_fraud,df_fraud_val):
    # Creating new features in training dataset
    has_null = df_fraud.isnull().sum(axis=1) != 0
    df_fraud['has_null'] = has_null
    df_fraud['account_lifetime'] = (
        (df_fraud["trx_date"].astype("datetime64[ns]")) - (df_fraud["registereddate"].astype("datetime64[ns]"))
        ).dt.days
    df_fraud['user_transaction_group'] = df_fraud[['centrality_indegree_p2p', 'centrality_outdegree_p2p']].apply(
        lambda row : assign_group(row['centrality_indegree_p2p'], row['centrality_outdegree_p2p'], 4),
        axis=1)
    df_fraud['is_group_1'] = df_fraud['user_transaction_group'] == 1
    df_fraud['is_group_2'] = df_fraud['user_transaction_group'] == 2
    df_fraud['is_group_3'] = df_fraud['user_transaction_group'] == 3

    # Creating new features in validation dataset
    df_fraud_val['has_null'] = has_null
    df_fraud_val['account_lifetime'] = (
        (df_fraud_val["trx_date"].astype("datetime64[ns]")) - (df_fraud_val["registereddate"].astype("datetime64[ns]"))
        ).dt.days
    df_fraud_val['user_transaction_group'] = df_fraud_val[['centrality_indegree_p2p', 'centrality_outdegree_p2p']].apply(
        lambda row : assign_group(row['centrality_indegree_p2p'], row['centrality_outdegree_p2p'], 4),
        axis=1)
    df_fraud_val['is_group_1'] = df_fraud_val['user_transaction_group'] == 1
    df_fraud_val['is_group_2'] = df_fraud_val['user_transaction_group'] == 2
    df_fraud_val['is_group_3'] = df_fraud_val['user_transaction_group'] == 3

def feature_encoding(df_fraud, df_fraud_val):
   #Label encoding
    mapping_gender = {
        'Female' : 0,
        'Male' : 1
    }

    df_fraud['gender(num)'] = df_fraud['gender'].map(mapping_gender)

    # Lakukan untuk dev
    df_fraud_val['gender(num)'] = df_fraud_val['gender'].map(mapping_gender)

    #One hot encoding
    for cat in ['job_position', 'source', 'user_transaction_group']:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc.fit(df_fraud[['job_position']])
        new_cols = [f"job_position_{job}" for job in enc.categories_[0]]

        # OHE for training data
        df_fraud[new_cols] = pd.DataFrame(enc.transform(df_fraud[['job_position']]),
                            columns=new_cols,
                            index = df_fraud.index)

        # OHE for validation data
        df_fraud_val[new_cols] = pd.DataFrame(enc.transform(df_fraud_val[['job_position']]),
                                columns=new_cols,
                                index = df_fraud_val.index)
        
    enc = OneHotEncoder(sparse_output=False)
    pd.DataFrame(enc.fit_transform(df_fraud[['job_position']]),
                columns=[f"job_position_{job}" for job in enc.categories_[0]],
                index = df_fraud.index
                )
    
def feature_scaling(df_fraud, df_fraud_val):
    scaled_cols = ['aqc_freq_prepaid_mobile', 'aqc_mean_prepaid_mobile_amount', 'aqc_freq_topup',
    'aqc_freq_topup_within_7d', 'aqc_mean_topup_amount_90d', 'aqc_total_topup_amount_7d',
    'aqc_freq_x2x_within_60d', 'aqc_mean_x2x_amount_90d', 'aqc_total_x2x_amount',
    'dormancy_max_gmt_pay_diff_days', 'dormancy_mean_gmt_pay_diff_days', 'dormancy_count_trx',
    'kyc_total_failed', 'kyc_total_revoked', 'avg_other_weight_1', 'centrality_outdegree_p2p',
    'centrality_indegree_p2p', 'centrality_outdegree_sendmoney']

    scalers_std = dict()

    for col in scaled_cols:
        scalers_std[col] = StandardScaler()

        # fit scaler and scale train
        df_fraud[f'std_{col}'] = scalers_std[col].fit_transform(df_fraud[col].values.reshape(len(df_fraud), 1))

        # # Also scale for test df
        df_fraud_val[f'std_{col}'] = scalers_std[col].transform(df_fraud_val[col].values.reshape(len(df_fraud_val), 1))

global MODELS_EVAL
MODELS_EVAL = dict()

def train_and_evaluate(X_train, y_train, X_val, y_val, model, model_name, description="no description", append=True, kwargs=None):
  """Doing train and evaluation in single run"""
  # train model
  model.fit(X_train, y_train)

  # predict on train
  y_pred = model.predict(X_train)

  # Evaluate Bias
  roc_auc_bias = roc_auc_score(y_train, y_pred)
  precision_bias = precision_score(y_train, y_pred)
  recall_bias = recall_score(y_train, y_pred)
  f1_bias = f1_score(y_train, y_pred)

  bias_score = {
                'roc_auc' : roc_auc_bias,
                'precision' : precision_bias,
                'recall' : recall_bias,
                'f1' : f1_bias}


  # predict on Validation
  y_pred = model.predict(X_val)

  # Evaluate Variance
  roc_auc_var = roc_auc_score(y_val, y_pred)
  precision_var = precision_score(y_val, y_pred)
  recall_var = recall_score(y_val, y_pred)
  f1_var = f1_score(y_val, y_pred)

  variance_score = {
                'roc_auc' : roc_auc_var,
                'precision' : precision_var,
                'recall' : recall_var,
                'f1' : f1_var }

  # Append scores to global variable
  model_eval = {
      'model' : model,
      'bias_score' : bias_score,
      'variance_score' : variance_score,
      'description' : description
    }

  if append:
    MODELS_EVAL[model_name] = model_eval
  
    # Print evaluation
    print(f" \
    EVALUATION OF : {model_name}\n\n \
    BIAS SCORES:\n \
        ROC_AUC   = {roc_auc_bias}\n \
        PRECISION = {precision_bias}\n \
        RECALL    = {recall_bias}\n \
        F1        = {f1_bias}\n \
    \n \
    VARIANCE SCORES:\n \
        ROC_AUC   = {roc_auc_var}\n \
        PRECISION = {precision_var}\n \
        RECALL    = {recall_var}\n \
        F1        = {f1_var}\n \
        \n \
    Description : {description}")

    return y_pred, precision_bias, precision_var

if __name__ == '__main__':
    # Run the Flask app
    app.run()
