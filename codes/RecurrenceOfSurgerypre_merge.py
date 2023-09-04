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