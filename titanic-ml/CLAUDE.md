# titanic-ml

## 목표
- 전통 ML(표 형태 데이터)을 직접 손코딩하며 이해하는 미니 프로젝트 (ML 전반을 넓게 배우는 계획의 1번)
- Titanic 데이터(seaborn 내장, data/titanic.csv)로 생존 예측

## 나(사용자)에 대해
- 한국어로 대화. pandas/scikit-learn은 써 봤지만 많이 잊은 상태
- 코드는 내가 직접 짠다. 정답 코드를 통째로 주지 말고, 쓸 함수 이름이나 힌트, 개념 설명으로 안내할 것
- 내가 쓴 코드와 결과를 보여 주면 틀린 곳만 짚어 주고, 이유를 스스로 설명해 보게 할 것
- 환경: Windows, PowerShell, .venv 사용

## 폴더 구조
data/ (git 제외), notebooks/01_eda.ipynb, src/{data,features,models,evaluate}.py, train.py, results/
- 먼저 노트북에서 탐색하고, 안정되면 src/*.py로 옮긴다

## 규칙
- train/test 분리를 가장 먼저 한다 (데이터 누수 방지). `alive` 열은 타깃과 같은 정보라서 feature에서 제외
- 결측치 채우기는 train 데이터로만 계산 (Pipeline 사용)
- 모든 실험은 baseline 대비 전/후 점수를 results/에 기록

## 진행 상황
- [x] 환경 세팅, 데이터 불러오기, shape/head/결측치 확인
- [ ] 생존자 비율 확인, DummyClassifier baseline
- [ ] Pipeline(ColumnTransformer), 모델 3개 비교, 지표(F1/AUC), 과적합 확인, feature 추가