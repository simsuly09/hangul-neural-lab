# 한글 신경망 실험실

한글 손글씨를 쓰면 실제 CNN의 중간 활성값과 예측 결과를 3D로 탐색하는 정적 웹앱이다. GitHub Pages에서 서버용 Python이나 별도 AI 서버 없이 실행한다.

웹사이트: https://simsuly09.github.io/hangul-neural-lab/

## 사용 방법

1. 웹사이트에 접속한다.
2. `모델 파일 2개 선택`을 누른다.
3. 앞서 받은 로컬 실행본의 `model` 폴더에서 `network.json`과 `weights.bin`을 함께 선택한다.
4. 한 글자를 쓰거나 기본 예시를 선택하고 인식한다.

`이 브라우저에 기억하기`를 켜면 모델을 해당 사이트의 IndexedDB에 저장하며 다음 접속 때 자동으로 불러온다. 파일을 서버로 업로드하지 않는다. 브라우저의 사이트 데이터를 삭제하면 다시 선택해야 한다.

선택할 파일은 AI Hub 원본 ZIP 자체가 아니라, 사용자에게 제공한 로컬 실행본에서 변환된 두 파일이다. 모델 사용 권한이 있는 사용자가 개인 브라우저에 불러와 사용하는 방식이다.

## 공개 파일 범위

이 저장소에는 직접 작성한 웹 코드, 코드로 그린 기본 예시, TensorFlow.js와 Three.js 및 라이선스만 포함한다. AI Hub 모델, 가중치, 손글씨 원본, 검증용 이미지와 라벨은 포함하지 않는다. 추론과 시각화는 사용자가 불러온 파일로 브라우저 안에서 수행한다.

## 기능

- 초성·중성·종성 3개 독립 CNN 분기 탐색.
- 전체 층 추론, 주요 층과 선택 층의 실제 활성값 3D 표시.
- 필터 이동과 실제 합성곱 수치, 최대 풀링, Flatten 및 Softmax 확인.
- 마우스 드래그로 회전, 휠로 확대, 오른쪽 드래그로 이동.
- 실제 계산 시간과 별개인 교육용 재생.

3D 표시에서는 일부 채널과 공간 위치를 샘플링한다. 화면의 안내와 연산 패널에서 생략 범위를 확인할 수 있다. 모델의 오인식도 그대로 표시한다.

## GitHub Pages 설정

저장소 Settings → Pages → Build and deployment에서 `Deploy from a branch`, `main`, `/(root)`를 선택해 저장한다. `.nojekyll`이 포함되어 있다. 모든 경로가 상대 경로이므로 저장소 이름 아래 경로에서도 실행된다.

로컬 확인은 `python -m http.server 8765 --bind 127.0.0.1` 실행 후 `http://127.0.0.1:8765/`에 접속하면 된다.

## 데이터 출처와 권리

이 프로젝트는 한국지능정보사회진흥원 AI Hub 사업 결과인 **다양한 형태의 한글 문자 OCR**의 모델을 개인 브라우저에서 활용하도록 설계했다.

- 소개: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=91
- 이용정책: https://aihub.or.kr/intrcn/guid/usagepolicy.do

원 제공기관의 모델과 데이터에는 해당 기관의 이용 조건이 적용된다. 이 저장소는 해당 파일을 재배포하지 않는다.

시각화 참고: https://github.com/okdalto/conv_visualizer. 기존 Processing 소스를 복사하지 않고 웹용 구현을 별도로 작성했다.

포함 라이브러리: TensorFlow.js 4.22.0 (Apache-2.0), Three.js 0.169.0 및 OrbitControls (MIT). 라이선스는 `vendor/`에 포함되어 있다.
