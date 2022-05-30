# MISOChallenge2021_infants_action_estimation

## 1. Overview
### 대회 개요.

- **목표 : 영유아 행동 영상 데이터를 활용하여 구현 가능한 AI 아이디어 제안 및 이를 간단히 구현한 AI 모델 제출**
  - Task : 행동 평가 코드(1~4 class) 예측 (*주최 측에서 정한 것이 아닌 직접 선정한 TASK)

- **데이터 : 영유아 대근육 행동(1-4번) 영상 및 행동 평가 결과 JSON 파일**
- **평가지표 :**
  - 아이디어 혁신성 : 참신성, 영유아 행동분류 활용 연관성
  - 상용화 가능성 : 구체성, 실용가능성
  - 기술적 타당성 : 개발과정, 논리적 타당성, AI 모델 결과 정확도 (accuracy)
- **대회기간 : 21.11.23 - 21.12.6**
- **주최 : NIA** 
- **주관 : (주)미소정보기술**
### 대회 성적
**rank : 최우수상 (2/6) 🏆️**.   
**tech : torch, torch-vision, numpy**


### 전체 프로세스 도식
-  자세한 내용은 대회 규정 등을 고려해 비공개.
-  직접 담당했던 내용*[papari1123](https://github.com/papari1123)에 대해서만 간략히 서술.
   

![image](https://user-images.githubusercontent.com/33012030/171026704-62b6fc29-3233-401b-b5e0-47a4602c2e69.png)



### 사용 모델 : Spatio-temporal Attention-based Model (STAM)
![image](https://user-images.githubusercontent.com/33012030/171026118-fb8a6efc-afe8-4949-b257-1ed99294afa4.png)


- Video Anomaly detection을 위해 설계된 Spatio-temporal Attention-based Model (STAM)
원 논문은 “비디오에서 뇌성마비 행동 징후를 탐지”하는 작업을 수행함
- Input: pose sequence (Skeleton graph) / Output: behavior evaluation class [0, 1, 2, 3]

## 2. 팀빌딩
**- Leader : [thomas11809](https://github.com/thomas11809)**     
  - Ph.D. student at Seoul National University (SNU). /B.S. in Department of ECE, SNU.      
  - **Role :** 포즈 예측 모델 적용, 논문 서베이, 데이터 라벨링 및 정제, 영유아 행동 평가 관련 아이디어 제시, 자료취합. 
 
**- Follower1 : Ph.D. student at Seoul National University (SNU).** 
  - **Role :** 행동 분류 모델 적용, 논문 서베이, 데이터 라벨링 및 정제, 영유아 행동 평가 관련 아이디어 제시, 자료취합. 
   
**- Follower2 : Ph.D. student at Seoul National University (SNU).**    
  - **Role :** 논문 서베이, 데이터 라벨링 및 정제, 영유아 행동 평가 관련 아이디어 제시, 자료취합. 

**- Follower3 : [papari1123](https://github.com/papari1123)**     
  - M.S. in Department of Human ICT convergence in SKKU. / B.S. in Department of Information Display, KHU.   
  - **Role:** 행동 분류 예측 모델 적용, 논문 서베이, 데이터 라벨링 및 정제, 영유아 행동 평가 관련 아이디어 제시, 자료취합. 



## @. reference
STAM(paper): https://nguyenthaibinh.github.io/papers/stam_jbhi.pdf     
STAM (URL): https://github.com/nguyenthaibinh/stam
