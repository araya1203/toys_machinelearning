# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# No 'Access-Control-Allow-Origin'
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

import pickle

# /api_v1/mlmodelwithregression with dict params
# method : post
@app.post('/api_v1/mlmodelwithregression') 
def mlmodelwithregression(data:dict) : # json
    print('data with dict {}'.format(data))
    # data dict to 변수 할당
    texture_mean = float (data['texture_mean'])
    perimeter_mean =float (data['perimeter_mean'])

    # pkl 파일 존재 확인 코드 필요

    result_predict = 0;
    # 학습 모델 불러와 예측
    with open('datasets/BreastCanceerWisconsin_Regression.pkl', 'rb') as regression_file:
        loaded_model = pickle.load(regression_file)
        input_labels = [[texture_mean, perimeter_mean]] # 학습했던 설명변수 형식 맞게 적용
        result_predict = loaded_model.predict(input_labels)
        print('Predict radius_mean Result : {}'.format(result_predict))
        pass

    # 예측값 리턴
    result = {'radius_mean':result_predict[0]}
    return result