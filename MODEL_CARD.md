# Model Card — CT porosity 검출 (배포용)

CT 단면 이미지에서 porosity(기공)를 찾아 셀을 `PASS` / `REJECT` 로 판정합니다.
서빙(`ai-infer`)이 이 문서만 보고 학습 결과를 재현할 수 있도록 쓴 계약 문서입니다.
출력 JSON 형식은 [`ct_output_schema.sample.json`](ct_output_schema.sample.json) 이 정본입니다.

| | |
| --- | --- |
| 최종 갱신 | 2026-08-03 |
| 모달 | CT (RGB/외관은 `model-rgb` 레포) |
| 판정 값 | `PASS` · `REJECT` (`FAIL` 은 촬영 품질 분류기 소관) |

---

## 1. 모델 신원

| 항목 | 값 |
| --- | --- |
| 가중치 | `ct_samedist_CHAMPION_ep6.pt` (S3 `models/` 프리픽스) |
| 아키텍처 | `yolo11m-seg` (instance segmentation, 사전학습 `yolo11m-seg.pt` 에서 파인튜닝) |
| 학습 run | `train_ct_tiled_v41_samedist` |
| 채택 에폭 | **6** (18에폭 중. `best.pt` 아님 — §4 conf 스윕 F1 으로 선택) |
| 메타 사이드카 | 가중치 옆 `ct_samedist_CHAMPION_ep6.json` (sha256·운영점 포함) |

> ⚠️ 사이드카의 `inference` 와 `operating_points` 는 **구 프로토콜(slice 1024 / overlap 0.4)** 에서 잰 값입니다.
> 배포 값은 이 문서와 스키마 샘플이 정본이고, 사이드카에는 `superseded_by` 블록으로 표시해 두었습니다.

### 클래스 매핑

| class_id | name | 의미 |
| --- | --- | --- |
| `0` | `porosity` | 기공 |

**단일 클래스입니다.** `data.yaml` 의 `names: ['porosity']` 가 유일한 출처이고, 출력 JSON의 `detections[].class` 에는
정수 id 가 아니라 이 문자열이 들어갑니다. 새 결함 유형을 임의로 만들지 않습니다(계약서 Core §6.5).

---

## 2. 추론 파이프라인

원본 CT 이미지는 세로로 매우 길고(최대 4000px) 결함은 수십 px 이라, 통짜로 리사이즈하면 신호가 사라집니다.
그래서 **폭은 네이티브로 두고 세로 스트립 타일로 학습**했고, 추론도 같은 크기로 잘라 넣습니다(SAHI).
**슬라이스 크기는 학습 타일 크기와 반드시 같아야 합니다** — 이게 어긋난 게 0728 이전 성능 저하의 원인이었습니다.

```
원본 ROI 이미지 (예: 562 x 4000)
  └ SAHI sliced inference   slice 1280 x 1280, overlap 0.2
      └ YOLOv11m-seg (conf 0.05)
          └ 슬라이스 결과 병합 (GREEDYNMM, match_threshold 0.5)
              └ detections[] + verdict(REJECT if len>0 else PASS)
```

### 추론 파라미터 (그대로 복사해 쓸 것)

```yaml
model:
  weight: ct_samedist_CHAMPION_ep6.pt
  task: segment
  classes: { 0: porosity }

sahi:
  enabled: true                     # CT 는 ON. RGB 는 OFF (계약서)
  slice_height: 1280                # = 학습 TILE. 바꾸지 말 것
  slice_width: 1280
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  postprocess_type: NMS              # 명시 필수 — 기본값은 conf 에 따라 바뀜
  postprocess_match_metric: IOU
  postprocess_match_threshold: 0.5

detection:
  confidence_threshold: 0.05        # 배포 운영점
```

### IoU · NMS — 실제로 두 층입니다

혼동이 잦아 분리해 적습니다. **코드가 명시적으로 지정하는 값은 SAHI 병합 임계값 하나뿐입니다.**

| 층 | 파라미터 | 값 | 출처 |
| --- | --- | --- | --- |
| ① 슬라이스 내부 NMS | `iou` | **미지정 → ultralytics 기본값 `0.7`** (8.4.115 실측) | `AutoDetectionModel.from_pretrained(...)` 에 안 넘김 |
| ① 슬라이스 내부 NMS | `max_det` | **미지정 → ultralytics 기본값 `300`** (8.4.115 실측) | 동일 |
| ② 슬라이스 간 병합 | `postprocess_match_threshold` | **0.5** | `get_sliced_prediction(...)` |
| ② 슬라이스 간 병합 | `postprocess_type` / `match_metric` | **`NMS` / `IOU`** (명시 지정) | 아래 주의 참고 |
| ② 슬라이스 간 병합 | `postprocess_class_agnostic` | **미지정 → sahi 기본값** (`False`). 단일 클래스라 영향 없음 | `get_sliced_prediction(...)` |
| ③ 평가 전용 | `IOU_HIT` | **0.1** | 채점용 localization 판정 임계값. **추론 경로와 무관** |

> ⚠️ **sahi 는 conf 가 낮으면 병합 방식을 조용히 바꿉니다.** 0.12.5 에서 배포 운영점 conf 0.05 로
> 돌리면 기본값 `GREEDYNMM`/`IOS` 대신 `NMS`/`IOU` 로 자동 전환되고 경고만 찍습니다
> (`Switching postprocess type/metric to NMS/IOU since model confidence threshold is low`).
> 그래서 코드에 `NMS`/`IOU` 를 **명시**했습니다 — 값은 그대로지만 conf 를 올려도 병합 방식이 안 바뀝니다.
> **conf 0.15(report_f1max)로 재평가하면 자동 전환이 안 걸려 배포와 다른 방식으로 병합됩니다.**
> 평가 노트북에서 conf 를 올려 잰 수치를 배포 수치와 직접 비교할 때 이 점을 감안하세요.
>
> ①의 "미지정" 항목은 **설치된 라이브러리 버전이 바뀌면 조용히 값이 바뀝니다.** 서빙에서는
> 기본값에 기대지 말고 명시적으로 넘겨 고정하세요. ③을 NMS 값으로 오해하지 마세요.

### 운영점 (conf)

| conf | 성격 | 실측 |
| --- | --- | --- |
| 0.02 | 최대 recall | R 0.990 · 재검부하 11.8% |
| **0.05** | **배포 기본값** | **R 0.977 · loc 86.7% · 재검부하 6.6%** |
| 0.10 | F1 최대 | F1 0.921 (모델 비교용, 배포용 아님) |

불량 게이트라 **recall 우선**입니다. 놓친 불량이 오탐보다 훨씬 비쌉니다 — 오탐은 사람 재검으로 회수되지만
미검출은 그대로 나갑니다. 위 숫자는 **slice 1280 / overlap 0.2 + epoch 6 조합 전용**이며, 다른 가중치나
다른 슬라이스 설정에서는 성립하지 않습니다.

---

## 3. 좌표계

납품 이미지 자체가 이미 ROI 입니다. `bbox_xyxy` 는 **그 이미지의 픽셀 좌표**이고 origin 은 좌상단입니다.
**좌표 변환·오프셋이 필요 없습니다.** 자세한 건 스키마 샘플의 `coordinate_space` 블록과
`battery_infer_to_json.ipynb` §4(자체검증 게이트)를 보세요.

CT 한 장은 단면이라 bbox 가 2축만 담습니다. 나머지 1축은 `cell.slice_index` 와 `volume.axis_mapping` 이 채웁니다.

---

## 4. 학습 재현

| 항목 | 값 |
| --- | --- |
| 데이터 | v4.1, `nobig` 정책 (`porosity_bbox_max_ratio ≥ 0.25` 인 대형 오라벨 이미지 제외) |
| split | **동분포(known-type)** — 셀 그룹분할이 아니라 이미지 단위 stratified, 47셀 전부 train·val 양쪽, 셀당 15% val |
| seed | 42 |
| 타일 | `TILE = ceil(max(1280, Wmax)/32)*32` → 실측 1280. 가로 1칸 · 세로 스트립, 폭 네이티브 보존 |
| 타일 라벨 | shapely 교집합 클립 — 스트립에 걸친 결함을 조각 라벨로 분배(중심 배정하면 걸친 스트립이 배경 오라벨이 됨) |
| 배경 타일 | `BG_KEEP 0.08`, `N_BG_TILES 1` |
| imgsz | `= TILE` (다운스케일 없음) |
| batch | 12 (OOM 시 8) |
| optimizer | AdamW, `lr0 0.0005`, `cos_lr`, `amp=True` |
| epochs / patience | 18 / 15 (채택은 6) |
| 증강 | `hsv_h 0.0, hsv_s 0.0, hsv_v 0.1, degrees 10, flipud 0.5, translate 0.2, scale 0.1, copy_paste 0.0` |
| seg 옵션 | `mask_ratio 2`, `overlap_mask False` |
| 학습 중 val | **`val=False`** — CT 는 mask mAP50 이 함정이라 학습 중 val 을 쓰지 않고, §4 conf 스윕 F1 으로 판정 |
| 학습 환경 | RunPod A40 / Colab |

노트북: `battery_ct_tiled_v41_samedist.ipynb` (§1 config → §2 타일 빌드 → §3 학습 → §4 평가 → §6 챔피언 백업)

---

## 5. 의존성 · 환경 — ⚠️ 현재 고정 안 되어 있음

노트북이 `!pip -q install ultralytics sahi shapely pandas pyyaml` 로 **버전 없이** 설치합니다.
즉 **오늘 돌린 결과와 다음 달 돌린 결과가 같다는 보장이 없습니다.** 서빙에 싣기 전에 반드시 고정하세요.

| 패키지 | 용도 | 현재 |
| --- | --- | --- |
| `ultralytics` | YOLOv11-seg 학습·추론, 슬라이스 내부 NMS | 미고정 |
| `sahi` | 슬라이스 추론·병합 | 미고정 |
| `torch` (+`torchvision`) | 런타임. CUDA 빌드가 GPU 와 맞아야 함 | 미고정 (Colab/RunPod 제공본) |
| `shapely` | 타일 라벨 클리핑 (**학습 전용**, 추론 불필요) | 미고정 |
| `pandas` · `pyyaml` | manifest·data.yaml | 미고정 |
| `pillow` | 이미지 I/O, 오버레이 | torch 의존으로 딸려옴 |

고정 절차 — 챔피언을 만든 그 런타임에서:

```bash
python -c "import ultralytics, sahi, torch; print(ultralytics.__version__, sahi.__version__, torch.__version__, torch.version.cuda)"
pip freeze > requirements.lock.txt
```

출력값을 아래 표에 적고 `requirements.lock.txt` 를 이 레포에 커밋하세요. **추측해서 채우지 마세요** —
버전이 틀리면 NMS 기본값이 달라져 같은 이미지에서 다른 박스가 나옵니다.

| | 버전 | 기록일 |
| --- | --- | --- |
| ultralytics | `8.4.115` | 2026-08-03 |
| sahi | `0.12.5` | 2026-08-03 |
| torch / CUDA | `2.11.0+cu128` / `12.8` | 2026-08-03 |
| torchvision | `0.26.0+cu128` | 2026-08-03 |
| numpy · pandas · pillow · shapely | `2.0.2` · `2.2.2` · `11.3.0` · `2.1.2` | 2026-08-03 |
| python | `3.12.13` | 2026-08-03 |

전체 목록은 `requirements.lock.txt`(Colab 런타임 freeze, 707줄)에 있습니다. 서빙 이미지에는 위 표의
패키지만 고정하면 충분하고, `shapely` 는 학습 전용이라 뺄 수 있습니다.

**기록 환경: Colab / Tesla T4.** 학습·서빙 GPU(A40 / EC2)와 다르므로, 골든 픽스처를 다른 GPU 에서
검증하면 부동소수 차이로 confidence 가 미세하게 흔들릴 수 있습니다.

---

## 6. 골든 픽스처 (회귀 테스트)

배포·리팩터링·라이브러리 업그레이드 후 **같은 입력에서 같은 출력이 나오는지** 확인하는 고정 표본입니다.

### 구성

```
fixtures/
  golden_ct.jpg           # 입력 이미지 1장 (결함 있는 것)
  golden_ct.json          # 위 추론 설정으로 낸 기대 출력
  golden_ct.sha256        # 입력 이미지 해시 (파일이 바뀌면 비교 자체가 무의미)
```

**기준 이미지는 "임계에서 가장 멀리 떨어진 검출"이 있는 장으로 고릅니다.** conf 가 임계(0.05)에
붙어 있는 케이스를 굳히면 라이브러리·GPU 가 조금만 달라져도 그 검출이 사라져 `verdict` 가 뒤집힙니다.
§5 가 자동으로 최고 confidence 케이스를 고르고, 여유가 0.05 미만이면 경고합니다.

### 생성 · 검증

`battery_infer_to_json.ipynb` **§5** 를 실행하면 `fixtures/` 4종(`golden_ct.jpg` · `golden_ct.json` ·
`golden_ct.sha256` · `env.json`)과 `requirements.lock.txt` 가 생성되고, 위 §5 표에 붙여넣을 버전 줄이
출력됩니다. 이 파일들을 레포에 커밋하세요.

픽스처는 **§2 가 실제로 낸 출력을 그대로 굳힙니다.** 픽스처용으로 따로 추론하면 배포 경로와 다른
코드를 검증하게 되므로 일부러 재추론하지 않습니다.

**§6** 이 회귀 검증입니다 — 같은 이미지에 다시 추론해 아래 허용오차로 비교하고, 어긋나면 `assert` 로
멈춥니다. 라이브러리 업그레이드·리팩터링 후 **§1 → §2 → §6** 순으로 돌리면 됩니다.

### 통과 기준

| 항목 | 허용 오차 |
| --- | --- |
| `num_detections` · `verdict` | **완전 일치** |
| `detections[].class` | **완전 일치** |
| `bbox_xyxy` | ±1.0 px |
| `confidence` | ±0.01 |
| `segmentation` | 폴리곤 점 개수가 달라질 수 있어 **면적 ±5%** 로만 비교 |

`verdict` 나 검출 개수가 달라지면 배포를 멈추세요. bbox 가 1px 넘게 움직이면 대개 라이브러리 버전이 바뀐 것입니다.

### 현재 픽스처 (2026-08-03)

| | |
| --- | --- |
| 이미지 | `CT__CT_cell_pouch_101_z_119__b82e0912.jpg` (셀 101, z축 119번, ROI 562x4000) |
| sha256 | `2f749661c58cee23a6a8cecf5ea195647e86c33411e34649e37035dbd6c2d97f` |
| 기대 출력 | `verdict REJECT` · 검출 1건 · `porosity` · conf **0.2705** · bbox `[269, 620, 274, 912]` |
| 임계 여유 | **+0.2205** (conf 0.05 기준) |
| 생성 환경 | Colab / Tesla T4 · ultralytics 8.4.115 · sahi 0.12.5 · torch 2.11.0+cu128 |
| 검증 | **§6 통과 확인 (2026-08-03)** — 회귀 테스트가 실제로 도는 것까지 확인됨 |

임계에서 0.22 떨어져 있어 라이브러리·GPU 가 조금 달라져도 검출이 사라지지 않습니다.
`bbox` 좌표는 sahi 가 정수로 내므로 서브픽셀 흔들림이 없고, 실질 변동은 `confidence` 와
`segmentation` 폴리곤뿐입니다.

---

## 7. 한계 · 알아야 할 것

- **새 셀 타입에서는 성능이 떨어집니다.** 위 운영점 수치는 학습에서 본 것과 같은 타입(동분포) 기준입니다.
  완전히 새로운 타입에서는 눈에 띄게 하락하며, 그 경우 재보정/파인튜닝이 필요합니다. 배포 시나리오를
  "같은 라인·같은 타입 재검사"로 볼지 "새 타입 유입"으로 볼지에 따라 기대치가 달라집니다.
- **mask mAP50 으로 이 모델을 평가하지 마세요.** 게이트가 하는 일은 마스크를 정확히 그리는 게 아니라
  불량 셀을 플래그하는 것이고, mAP 기준으로 고르면 실제로 더 나은 모델을 버리게 됩니다. 판정은
  image-level PASS/REJECT F1 과 localization-aware F1 로 합니다.
- **슬라이스·conf·가중치는 한 세트입니다.** 셋 중 하나만 바꾸면 위 실측값은 무효입니다.
- 물리 단위(mm)는 제공하지 않습니다. 원본에 voxel spacing 메타데이터가 없어 픽셀/복셀 단위까지만 나갑니다.
- `volume.slice_scale_px` 는 슬라이스가 ROI 를 균등히 덮는다는 가정의 근사값입니다.

---

## 8. 관련 파일

| 파일 | 역할 |
| --- | --- |
| `ct_output_schema.sample.json` | 출력 JSON 계약 (정본) |
| `battery_infer_to_json.ipynb` | 추론 → JSON + 오버레이 + 좌표계 자체검증 |
| `battery_ct_tiled_v41_samedist.ipynb` | 학습 파이프라인 · 챔피언 백업 |
| `battery_ct_tiled_v41_samedist_score_check.ipynb` | conf 스윕 점수 검증 |
