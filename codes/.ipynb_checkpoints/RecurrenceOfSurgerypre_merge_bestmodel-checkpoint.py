# #### quest
#  - 업무분장(전처리, 모델학습)
#  - RecurrenceOfSurgery.csv 사용
#  - 목표변수 범주형 : 척추전방위증 
#  - 설명변수 최소 6개 : 
#    - (연속형)
#      체중 
#      신장
        
#    - (범주형)
#      환자통증정도,
#      스테로이드치료, 
#      수술기법
#      과거수술횟수
  
# - 서비스 대상과 목표 설명, 변수 선택 이유 
import pandas as pd

df_ROS = pd.read_csv('../datasets/RecurrenceOfSurgery.csv')
df_ROS.info()

df_ROS[:2]

df_ROS.columns

df_ROS_extract_null = df_ROS[['척추전방위증', '체중', '신장', '환자통증정도', '스테로이드치료', '수술기법', '과거수술횟수']]
df_ROS_extract_null.isnull().sum() # 결측치 확인 

df_ROS_extract_null[:2]

df_ROS_extract_null['수술기법'].value_counts()

### 결측치 예측 훈련 모델 생성['수술기법']
from sklearn.linear_model import LogisticRegression

# 원본 데이터프레임에서 '수술기법' 열을 추출
df_ROS_extract_null['수술기법']

# 결측치를 예측하는 함수
def predict_surgery_method(row):
    if pd.isnull(row['수술기법']):
        # 결측치를 예측하기 위해 사용할 특성 선택 (이 예제에서는 다른 열을 모두 사용)
        features = df_ROS_extract_null.columns.tolist()
        features.remove('수술기법')  # 예측에 사용할 특성에서 '수술기법' 열 제외
        X = df_ROS_extract_null.loc[row.name, features].values.reshape(1, -1)
        
        # 학습 데이터에서 수술기법이 결측치인 행 제외
        df_train = df_ROS_extract_null.dropna(subset=['수술기법'])
        
        # 훈련용 데이터셋과 타겟 생성
        features_train = df_train[features].values
        target_train = df_train['수술기법'].values
        
        # 로지스틱 회귀 모델 훈련 (결측치 예측, 수술기법의 결측치를 대체하기 위한 용도)
        missing_value_imputation_model = LogisticRegression()
        missing_value_imputation_model.fit(features_train, target_train)
        
        # 결측치 대체를 위한 예측
        predicted_nan = missing_value_imputation_model.predict(X)
        
        # 예측 결과를 반환
        return predicted_nan[0]
    else:
        return row['수술기법']
    
    # apply를 사용하여 함수 적용하여 결측치 처리
df_ROS_extract_null.loc[:, '수술기법'] = df_ROS_extract_null.apply(lambda row: predict_surgery_method(row), axis=1)

df_ROS_extract_null.isnull().sum() 

df_ROS_extract_null['수술기법'].value_counts()

df_ROS_extract_null[:2]

# pip install imbalanced-learn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import TomekLinks 

# 수술기법이 결측치인 행 추출
nan_rows = df_ROS_extract_null[pd.isnull(df_ROS_extract_null['수술기법'])]

# 결측치가 아닌 행 추출
not_nan_rows = df_ROS_extract_null.dropna(subset=['수술기법'])

# Features와 Target 설정
features = not_nan_rows.drop(columns=['수술기법'])
target = not_nan_rows['수술기법'] 

# Tomek's Link 언더샘플링을 적용
tl = TomekLinks(sampling_strategy='majority')
features_resampled, target_resampled = tl.fit_resample(features, target)

# 데이터 분할 (70% 학습, 30% 테스트)
features_train, features_test, target_train, target_test = train_test_split(features_resampled, target_resampled, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습 (모델 평가, 결측치 예측 모델 평가를 위해 사용)
Evaluation_model = LogisticRegression()
Evaluation_model.fit(features_train, target_train)

# 모델을 사용하여 결측치 예측
predicted_nan_values = nan_rows.apply(predict_surgery_method, axis=1)


# 결측치 예측 결과를 원본 데이터프레임에 할당
df_ROS_extract_null.loc[nan_rows.index, '수술기법'] = predicted_nan_values

# 결측치 예측 모델 평가 accuracy 
target_pred = Evaluation_model.predict(features_test)
accuracy = accuracy_score(target_test, target_pred)
print("결측치 예측 모델 정확도:", accuracy)

# 결측치 예측 모델 평가 - F1 점수 계산
target_pred = Evaluation_model.predict(features_test)
f1 = f1_score(target_test, target_pred, average='weighted')
print("결측치 예측 모델 F1 점수:", f1)


### 신장, 체중 이상치 제거

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 한글 폰트를 지정
font_path = "C:/Windows/Fonts/malgun.ttf" 
font_prop = fm.FontProperties(fname=font_path, size=14)

# 그래프에서 한글을 사용할 때 폰트 설정
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False 

# 신장 데이터에 대한 상자 그림
plt.boxplot(df_ROS_extract_null['신장'])
plt.title('신장 이상치')
plt.ylabel('Height (cm)')
plt.show()

# 체중 데이터에 대한 상자 그림
plt.boxplot(df_ROS_extract_null['체중'])
plt.title('체중 이상치')
plt.ylabel('Weight (kg)')
plt.show()

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return df_cleaned 

# "신장" 열의 이상치 제거
df_ROS_extract_null = remove_outliers(df_ROS_extract_null, '신장')

# "체중" 열의 이상치 제거
df_ROS_extract_null = remove_outliers(df_ROS_extract_null, '체중')

plt.boxplot(df_ROS_extract_null['신장'])
plt.title('신장 이상치 제거')
plt.ylabel('Height (cm)')
plt.show() 

# 체중 데이터에 대한 상자 그림
plt.boxplot(df_ROS_extract_null['체중'])
plt.title('체중 이상치 제거')
plt.ylabel('Weight (kg)')
plt.show() 


### Scaling & Encoding 

df_ROS_extract_null.value_counts()

df_ROS_extract_null.value_counts()

### OneHotEncoding : 수술기법  범주형 데이터를 수치형 데이터로 변환

from sklearn.preprocessing import OneHotEncoder 

oneHotEncoder = OneHotEncoder()
oneHotEncoder.fit(df_ROS_extract_null[['수술기법']])  

columns_name = oneHotEncoder.categories_

# oneHotEncoder.transform(df_TFD_extract_preprocess[['Pclass']]).toarray() # 실제값 확인용
encoded_data = oneHotEncoder.transform(df_ROS_extract_null[['수술기법']]).toarray()

encoded_data.shape 

# 병합 위해 numpy array to DataFrame
df_encoded_data = pd.DataFrame(data=encoded_data, columns=oneHotEncoder.get_feature_names_out(['수술기법']))
df_encoded_data[:2] 

df_ROS_extract_encoded = pd.concat([df_ROS_extract_null.reset_index(drop=True)
                                       , df_encoded_data.reset_index(drop=True)], axis=1)
df_ROS_extract_encoded[:2] 

df_encoded_data.index, df_encoded_data.shape 

df_ROS_extract_null.index, df_ROS_extract_null.shape 

df_ROS_extract_encoded.shape 

df_ROS_extract_encoded.columns 

target = df_ROS_extract_encoded[['척추전방위증']] 

features = df_ROS_extract_encoded.drop(columns=['척추전방위증','수술기법'])

features.columns

### MinMaxScaler 

from sklearn.preprocessing import MinMaxScaler

minMaxScaler = MinMaxScaler() # 인스턴스화
features = minMaxScaler.fit_transform(features)
features.shape 


# Split
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=111)
features_train.shape, target_train.shape, features_test.shape, target_test.shape

#### 모델학습

from sklearn.tree import DecisionTreeClassifier # 연속형 변수와 관련된 작업 = Tree


model = DecisionTreeClassifier()

from sklearn.model_selection import GridSearchCV

hyper_params = {'min_samples_leaf' : [5, 7, 9]
               ,'max_depth' : [9, 11]
               ,'min_samples_split' : [5, 6, 7]}

#### 평가 Score Default, 분류(Accuracy), 예측(R square)

# 평가 score 분류
from sklearn.metrics import f1_score, make_scorer

scoring = make_scorer(f1_score) 


grid_search = GridSearchCV(model, param_grid=hyper_params, cv=3, verbose=1, scoring=scoring) 

grid_search.fit(features_train, target_train)

grid_search.best_estimator_ 

grid_search.best_score_, grid_search.best_params_ 

best_model = grid_search.best_estimator_ # 하나의 모델 --> 그 중에서 최고의 모델
best_model  

target_test_predict = best_model.predict(features_test)
target_test_predict 

# 평가
from sklearn.metrics import classification_report

print(classification_report(target_test, target_test_predict))
