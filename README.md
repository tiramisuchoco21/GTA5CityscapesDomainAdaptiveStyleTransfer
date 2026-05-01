# # 인공지능응용및실습
## [자료_기본]
[인응실_결과물_자료.zip](https://github.com/user-attachments/files/27111734/_._.zip)
- 인응실_발표자료_최종
- 인응실_보고서_최종

## [자료_추가]
- #### 기존 코드에 다른 팀에서 적용한 방법 중 기존 결과에서 Fid 값이 더 낮도록 추가하여 모델 학습한 결과입니다.
- 팀프로젝트 당시 사용한 GPU 환경과 달라서(더 제한적임) 비교를 위해 가능한 환경에서 같은 조건으로 진행하였습니다.
- ProCST/core에 추가한 models_AdaLIN.py 업로드

> mIoU 고정
<img width="202" height="41" alt="image" src="https://github.com/user-attachments/assets/274f0ba5-c0ca-4d7d-ab27-49d3ef66b000" />

> 기존 결과
<img width="219" height="65" alt="image" src="https://github.com/user-attachments/assets/4f0a90ca-0e2a-4d22-be08-e5e6ebe893fa" />

> 추가 결과
<img width="214" height="65" alt="image" src="https://github.com/user-attachments/assets/580341c9-ed2b-47a3-a8cf-f4da86ee72b4" />
<img width="256" height="128" alt="image" src="https://github.com/user-attachments/assets/de1633fa-a435-4a4e-a31b-3a510885492a" />

> 정리
- 추가 결과에는 AdaLIN Generator를, Discriminator은 기존 ProCST 코드를 사용하였습니다.
- AdaLIN의 역할은 이미지의 기하학적 구조를 유지하면서 스타일을 입혀 유연성을 높여줍니다.
- 실험한 결과, 추가 결과는 기존 결과에 비해 FID 수치가 4.6정도 낮아졌습니다.
- 드라마틱하게 낮아질 것이라 생각했지만, AdaLIN이 스타일 변환으로 GTA와 Cityscapes의 차이를 부드럽게 해주더라도, Discriminator가 잡아내는 기준이 엄격하다면 Generator가 과감하게 FID를 낮출 만큼 변하지 못합니다.
- 이러한 점에서 시각적으로 어색한 부분을 줄이거나 경계선을 깔끔하게 만드는 데 집중되었을 것이라 예상합니다.
