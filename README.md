RecurrenceOfSurgery 

👉 팀원 : 김민정 , 조아라 

|    | 서비스 대상         | 변수 설정 이유                                                                                   |
|---:|:----------------|:----------------------------------------------------------------------------------------------------------|
|    |  의료 기관 (병원) |  이 프로젝트의 목표는 환자의 척추 전방전위증의 발생 여부를 예측하는 머신러닝 모델을 개발하여 의료 기관에 제공하는 것임. 척추 전방전위증은 환자의 척추 상태와 관련된 중요한 진단 중 하나로, 정확한 예측은 환자의 치료 및 관리에 도움이 되도록 전방전위증의 발생 여부를 예측하고, 환자의 치료 방향을 지원함.


|    | 목수변수명 (범주형)           | 변수 설정 이유                                                                         |
|---:|:----------------|:----------------------------------------------------------------------------------------------------------|
|    |  척추전방전위증   |척추 상태와 관련된 중요한 진단 중 하나 |

|    | 설명변수명 (범주형)           | 변수 설정 이유                                                                                   |
|---:|:----------------|:----------------------------------------------------------------------------------------------------------|
| 1  | 체중(Weight)    | 체중이 무거울수록 척추에 가해지는 압력이 더 높아질 수 있으며, 이는 전방위증의 발생 가능성을 높일 수 있음 |
| 2  | 신장(Height)    | 환자의 신장이 짧을수록 척추에 가해지는 압력이 다를 수 있으므로 이 변수는 예측 모델에 영향을 미칠 수 있음 |
| 3  | 입원기간        | 척추전방위증 환자의 치료 및 회복 기간을 나타내며, 입원 기간이 길면 병원 비용 및 환자의 불편함(-)이 증가하지만 병원에는 (+)이득으로 보임 |

|    | 설명변수명 (범주형)           | 변수 설정 이유                                                                                   |
|---:|:------------------|:---------------------------------------------------------------------------------------|
| 4  | 환자통증정도      | 통증 정도가 높을수록 환자가 치료 및 회복 과정에서 추가적인 관리와 주의가 필요함     |
| 5  | 스테로이드치료    | 스테로이드 치료의 적정한 사용은 안전성과 효과를 극대화함                            |
| 6  | 수술기법          | 수술 기법은 환자의 치료 및 회복에 큰 영향을 미칠 수 있으며, 다양한 수술 기법에 따라 결과가 다름 |
| 7  | 과거수술횟수      | 과거에 수술을 더 많이 받은 환자들이 척추전방위증 발생 가능성이 높을 수 있음          |









