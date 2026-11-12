# eip-jetson-bundle

Speck 칩에서 돌린 CIFAR10-DVS 9층 VGGSNN 을 Jetson Orin Nano 에서 **같은 모델·같은 1000장**으로 돌려 정확도·처리 시간·전력·에너지를 재기 위한 묶음.
Jetson 에는 numpy 와 TensorRT (JetPack 기본) 만 있으면 된다. torch, onnx, onnxruntime, pycuda 는 필요 없다. 스크립트는 sudo 를 쓰지 않는다.

## 구성

```
manifest.json            모델 12개 목록 (T, 체크포인트 해시, 원래 모델 정확도)
models/                  ONNX 36개 = 12 모델 x {b1: 배치 1, b16: 배치 16, b1_spk: 배치 1 + 층별 스파이크 합 출력}
data/frames_T8.npz       입력 1000장 (T=8 판), x uint8 [1000,8,2,32,32], y [1000]
data/frames_T16.npz      입력 1000장 (T=16 판)
data/expected_<모델>.npz 원래 (sinabs) 모델의 로짓·예측·층별 스파이크 합, 1000장 (Jetson 결과와 대조용)
jetson/build_engines.py  ONNX -> TensorRT 엔진 (FP32, FP16)
jetson/bench.py          정확도, 지연, 처리량, 층별 스파이크 대조, 전력·에너지
jetson/eipcommon.py      공통 (CUDA 런타임은 ctypes 로 직접 호출)
tools/build_onnx.py      ONNX 생성기 (참고용, Jetson 에서는 쓰지 않음)
```

## 모델과 입력

- 구조: Speck 9층 VGGSNN (`model_variant=pyramid_late64`, 32x32 입력 2채널, conv0..conv7 + fc, 층마다 IAF, conv0 뒤 BN, 풀링은 SumPool 2x2 3번).
  IAF 는 sinabs 와 같은 식이다: 막전위에 입력을 더하고, 스파이크 = floor(max(v, 0) / 1.0), 빼기 리셋 (v -= 스파이크), 하한 -1.0 (`v = relu(v + 1) - 1`). 한 단계에 여러 스파이크 가능 (정수 개수).
- ONNX 는 T 단계를 펼친 순수 텐서 그래프 (opset 17). 컨볼루션·풀링은 [B*T, C, H, W] 로 한 번에, IAF 만 T 단계로 펼친다 (수식은 단계별 계산과 같다).
  입력 `input` [B, T, 2, 32, 32] float32, 출력 `logits` [B, 10] (T 평균), b1_spk 판은 `layer_spikes` [B, 9] (층별 출력 스파이크 합, T 전체, conv0..conv7, fc) 도 낸다.
- 입력 1000장은 Speck 칩 B 측정과 **같은 표본** (CIFAR10-DVS 검증 분할 1000장, 칩 측정 표본 목록의 label 과 표본별 입력 이벤트 수가 1000장 모두 일치함을 확인).
  전처리: 좌표 //4 (128 -> 32), 이벤트 5개마다 1개 (speck_subsample_ratio 5), 시간 구간을 T 등분해 이벤트 개수를 셈. 값은 0~42 정수 (uint8). T=8 모델은 `frames_T8`, T=16 모델은 `frames_T16`.
  칩 측정 입력에는 상수 채널 (이미지당 T x 1024 이벤트) 이 추가로 들어갔다. 여기서는 그 채널 없이 BN 을 그대로 둔 SW 경로 (`count_silent` 와 같은 계산, 칩 경로 SW) 를 따른다.

| 이름 | method | T | 원래 모델 정확도 % (1000장) | ONNX 정확도 % (canus onnxruntime) | 예측 일치 | 이미지당 스파이크 합 (원래 모델) |
|---|---|---|---|---|---|---|
| `ours_T16_s42` | ours (λ 상한 1.25e-4) | 16 | 59.5 | 59.5 | 1000/1000 | 48,898 |
| `base_T8_s42` | baseline | 8 | 58.5 | 58.5 | 1000/1000 | 64,279 |
| `sip2e8_T16_s42` | SIP λ2e-8 | 16 | 58.5 | 58.5 | 1000/1000 | 31,564 |
| `ours_T16_s43` | ours (λ 상한 1.25e-4) | 16 | 57.4 | 57.4 | 1000/1000 | 34,744 |
| `base_T8_s43` | baseline | 8 | 60.3 | 60.3 | 1000/1000 | 48,945 |
| `sip2e8_T16_s43` | SIP λ2e-8 | 16 | 58.4 | 58.4 | 1000/1000 | 27,416 |
| `ours_T16_s44` | ours (λ 상한 1.25e-4) | 16 | 59.1 | 59.1 | 1000/1000 | 43,982 |
| `base_T8_s44` | baseline | 8 | 59.6 | 59.7 | 999/1000 | 60,196 |
| `sip2e8_T16_s44` | SIP λ2e-8 | 16 | 58.7 | 58.7 | 1000/1000 | 27,801 |
| `ours_T16_s45` | ours (λ 상한 1.25e-4) | 16 | 58.0 | 58.0 | 1000/1000 | 29,626 |
| `base_T8_s45` | baseline | 8 | 59.8 | 59.8 | 1000/1000 | 48,838 |
| `sip2e8_T16_s45` | SIP λ2e-8 | 16 | 58.6 | 58.6 | 1000/1000 | 25,137 |

방법별 평균 (n=4 시드): ours T=16 58.50, baseline T=8 59.55 (ONNX 59.58), SIP T=16 58.55 (1000장 기준).
논문의 SW 정확도 (ours 58.75, baseline 59.35) 와 숫자가 다른 이유: 논문 값은 학습 로그의 검증 992장 (배치 16 x 62) 이고 학습 중 기록한 값이다 (장치·수치 경로 차이는 확인하지 못함). 여기 값은 같은 체크포인트를 CPU 로 1000장 전부 돌린 값이다 (앞 992장만 세면 ours 58.62, baseline 59.68, SIP 58.70).

## 실행 순서

0. (선택, 사용자가 직접) 전원 모드와 클록을 고정하고 기록한다. 예: `sudo nvpmodel -m <모드>`, `sudo jetson_clocks`. 스크립트는 `nvpmodel -q`, `jetson_clocks --show` 를 sudo 없이 시도해 결과에 적는다 (권한이 없으면 "실행 못함" 으로 남는다).
   측정 중에는 다른 프로세스를 두지 말고, 모델 비교 내내 같은 모드를 쓴다.
1. 저장소를 받는다 (`git clone` 후 폴더로 이동). 비공개 저장소이므로 토큰 인증이 필요하다. 어려우면 GitHub 웹에서 ZIP 으로 받아도 된다.
2. 동작 점검 (모델 1개, 100장, 몇 분):
   ```
   python3 jetson/build_engines.py --models ours_T16_s42 --variants b1 b1_spk --precisions fp32
   python3 jetson/bench.py --models ours_T16_s42 --precisions fp32 --tasks acc latency spikes --n 100
   ```
   FP32 에서 `원래모델과 예측 일치` 가 100장 중 99~100 이어야 정상이다.
3. 엔진 빌드 (전체: 12 모델 x 3 변형 x 2 정밀도 = 72개, 수십 분~수 시간. 타이밍 캐시를 공유해 뒤로 갈수록 빨라진다):
   ```
   python3 jetson/build_engines.py
   ```
   줄여서: `--models ours_T16_s42 base_T8_s42 sip2e8_T16_s42 --variants b1 b16 b1_spk --precisions fp32 fp16`
4. 정확도·지연·처리량·스파이크 대조 (전력 제외, 모델 12개 전부):
   ```
   python3 jetson/bench.py --tasks acc latency throughput spikes
   ```
5. 전력·에너지 (모델당 정밀도당 (유휴 60초 + 처리 30초 이상) x 3회 x 배치 2종 = 약 10분. 대표 모델부터 권장):
   ```
   python3 jetson/bench.py --models ours_T16_s42 base_T8_s42 sip2e8_T16_s42 --tasks acc power
   ```
   전부 한 번에 돌리려면 `python3 jetson/bench.py` (오래 걸림). 결과는 `results/bench_<시각>.json`, 요약 표는 화면에 나온다. 나중에 표만 다시 보려면 `python3 jetson/bench.py --summarize results/<파일>.json`.

옵션: `--idle_s 60` `--min_run_s 30` `--power_repeats 3` `--warmup 50` `--power_batches 1 16`.

## 측정 정의 (Speck 표 2 와 같은 틀)

- **정확도**: 배치 1 엔진으로 1000장, 로짓 argmax. 같은 표본에서 원래 모델 예측과 몇 장이 일치하는지도 적는다.
- **지연 (배치 1)**: 준비 50회를 버리고 1000장을 한 장씩. 표본당 시간 = H2D 복사 + 실행 + D2H 복사 + 동기화 (입력은 미리 float32 로 pinned 메모리에 올려 두어 데이터 읽기·형변환은 제외). 평균·중앙·p95·p99. 커널 실행만의 GPU 시간 (cudaEvent) 도 따로 적는다.
- **처리량 (배치 16)**: 1000장을 1008장으로 채워 (앞 8장 되풀이) 63 배치, 3회 통과의 평균. ms/표본, 표본/s.
- **전력**: INA3221 (`/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/`) 의 레일 VDD_IN (보드 전체), VDD_CPU_GPU_CV, VDD_SOC 를 별도 스레드가 약 10 ms 간격으로 읽는다. 전력 = 전압 (mV) x 전류 (mA) / 1000 [mW].
  한 번의 측정 = 유휴 60초 -> 1000장 연속 처리 (30초 이상 되풀이), 3회 반복. 배치 1 과 배치 16 을 각각 잰다.
- **동적 전력** = 처리 구간 평균 - 유휴 평균 (각 구간 맨 앞 1초는 버림). **표본당 에너지 [mJ]** = 동적 전력 [mW] x 처리 표본당 시간 [ms] / 1000. 레일별과 보드 전체 (VDD_IN) 모두 기록. 유휴를 빼지 않은 전체 전력 기준 에너지도 `energy_total_mJ_per_sample` 로 함께 있다.
- **층별 스파이크 대조**: `b1_spk` 엔진의 층별 스파이크 합을 `expected_*.npz` 와 비교 (이미지당 합의 차, 전 층이 정확히 같은 표본 수).

## 검증 결과 (canus, onnxruntime CPU)

- 원래 sinabs 모델 대 ONNX, 12개 x 1000장: 예측 일치 11개는 1000/1000, 1개 (`base_T8_s44`) 는 999/1000 (정확도 59.6 -> 59.7). 배치 16 판도 같은 결과.
- 층별 스파이크 합: 12,000 표본 중 68 표본에서 어느 한 층의 합이 다르고 (최대 차 41 스파이크), 이미지당 합의 평균은 소수 둘째 자리까지 같다.
  원인은 컨볼루션 커널의 부동소수 합산 순서 (onnxruntime 대 torch): 같은 계산을 torch 로 ONNX 와 같은 순서로 쓰면 원래 모델과 로짓·스파이크가 비트 단위로 일치한다 (3개 모델 확인).
  임계값 (floor) 경계에 걸린 뉴런이 한 개 뒤집히는 정도의 차이이고, **TensorRT 에서도 같은 크기의 차이가 나는 것이 정상**이다. FP16 은 정밀도가 낮아 더 많이 다를 수 있다 (그 차이도 결과에 기록된다).

## 한계

- TensorRT 엔진 빌드·실행 코드는 canus 에서 실행해 보지 못했다 (GPU 사용 금지, Jetson 접속 불가). 같은 스크립트 논리 (정확도·지연·처리량·스파이크 대조·전력 계산) 는 onnxruntime 백엔드와 가짜 센서 폴더로 점검했다. 처음 Jetson 에서 돌릴 때는 위 2번 점검부터 한다.
- Jetson 은 프레임 기반 밀집 연산이라 스파이크 수가 줄어도 시간·에너지가 줄지 않는다 (Speck 은 이벤트 기반이라 줄어든다). 비교 시 이 차이를 전제로 읽는다.
