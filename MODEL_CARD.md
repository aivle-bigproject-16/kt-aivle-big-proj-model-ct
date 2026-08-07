# Model Card — YOLO CT (porosity 검출)

CT 단면 이미지에서 porosity(기공)를 찾아 셀을 `PASS` / `REJECT` 로 판정합니다.
`ai-infer` 가 이 문서만 보고 학습 결과를 재현할 수 있도록 쓴 계약 문서입니다.

- 기계가 읽을 형태 → [`ct.model_card.json`](ct.model_card.json)
- 출력 JSON 형식 → [`ct_output_schema.sample.json`](ct_output_schema.sample.json)

| | |
| --- | --- |
| 최종 갱신 | 2026-08-04 |
| 모달 | CT (RGB/외관은 `model-rgb` 레포) |
| 판정 값 | `PASS` · `REJECT` (`FAIL` 은 촬영 품질 분류기 소관) |

---

## 0. 요청 항목 대조표

| 요청 (§) | 상태 | 값 / 위치 |
| --- | --- | --- |
| 1.1 식별 | 부분 | §1 — `weight_sha256` · `trained_at` 미기입 |
| 1.2 라이브러리 버전 | ✅ | §2 + `requirements.lock.txt` |
| 1.3 입력 전처리 | ✅ | §3 — 실효 `imgsz` **1280** (0806 정정 · 0807 백엔드 런타임 확인) |
| 1.4 추론 파라미터 | ✅ | §4 |
| 1.5 출력 매핑 | ✅ | §5 — `porosity` → `MICRO_DEFECT` |
| 2.2 SAHI 후처리 | ✅ | §4 — `NMS` / `IOU` / `0.5` 명시 |
| 2.3 사이드카가 낡음 | ✅ | §1 — 사이드카에 `superseded_by` 표시 완료 |
| 2.3 conf ↔ weight 짝 | ✅ | §6 — `0.05` 는 `samedist ep6` 전용 |
| 5 골든 픽스처 | 부분 | §7 — **1장**. 요청은 20~30장 |
| 6 model_card.json | ✅ | `ct.model_card.json` |
| 7 confidence 산식 | 답변 | §10 |
 
---

## 1. 식별

| 항목 | 값 |
| --- | --- |
| `weight_file` | `ct_samedist_CHAMPION_ep6.pt` |
| `weight_sha256` | **미기입** — 가중치 옆 사이드카 `ct_samedist_CHAMPION_ep6.json` 에 있습니다 |
| `architecture` | `yolo11m-seg` (instance segmentation, `yolo11m-seg.pt` 파인튜닝) |
| `train_run` | `train_ct_tiled_v41_samedist` |
| 채택 에폭 | **6** (18에폭 중. `best.pt` 아님 — §6 conf 스윕 F1 으로 선택) |
| `source_commit` | `dbfa3c54f40724ef612e5f57584ea506e582cfc8` (이 레포) |
| `trained_at` | **미기입** |

> ⚠️ **가중치 옆 사이드카를 읽지 마세요.** `ct_samedist_CHAMPION_ep6.json` 의 `inference` ·
> `operating_points` 는 **구 프로토콜(slice 1024 / overlap 0.4)** 값입니다. 그대로 쓰면 계약이 요구하는
> slice 1280 이 아니라 1024 로 돌게 됩니다. 사이드카에는 `superseded_by` 블록을 넣어 두었고,
> **정본은 이 문서와 `ct.model_card.json`** 입니다.

---

## 2. 라이브러리 버전

| 패키지 | 버전 | 서빙 필요 |
| --- | --- | --- |
| `python` | `3.12.13` | ✅ |
| `torch` / CUDA | `2.11.0+cu128` / `12.8` | ✅ |
| `torchvision` | `0.26.0+cu128` | ✅ |
| `ultralytics` | `8.4.115` | ✅ |
| `sahi` | `0.12.5` | ✅ |
| `numpy` · `pillow` | `2.0.2` · `11.3.0` | ✅ |
| `shapely` · `pandas` · `pyyaml` | `2.1.2` · `2.2.2` | ❌ 학습 전용 |

전체 목록은 [`requirements.lock.txt`](requirements.lock.txt) (Colab 런타임 freeze, 707줄).
**기록 환경: Colab / Tesla T4.** 서빙 GPU 가 다르면 부동소수 차이로 confidence 가 미세하게 흔들립니다.

버전을 바꾸면 §4 의 "미지정 → 기본값" 항목이 조용히 달라집니다. 올린 뒤에는 §7 회귀 테스트를 반드시 도세요.

---

## 3. 입력 전처리

| 항목 | 값 |
| --- | --- |
| `input_type` | **파일 경로** (sahi 가 직접 읽습니다) |
| `color_space` | `RGB` |
| `imgsz` | **1280** ← 아래 주의 |
| `letterbox` | `true` (ultralytics 기본) |
| `normalize` | ultralytics 기본 (0-1 스케일, mean/std 정규화 없음) |
| ROI | 납품 이미지 자체가 ROI. 별도 crop 불필요 |
| ROI 크기 고정? | **아니오** |

> ### ⚠️ 실효 `imgsz` 는 **1280** 입니다 (2026-08-06 정정)
>
> **이전 카드는 `640` 이라고 적었습니다. 틀렸습니다.** 아래가 실제 경로입니다.
>
> | | 확인한 것 |
> | --- | --- |
> | ① 체크포인트 | `ct_samedist_CHAMPION_ep6.pt` → `train_args.imgsz = 1280` |
> | ② ultralytics 8.4.115 | `_reset_ckpt_args` 의 `include = {"imgsz", "data", "task", "single_cls"}` — **`imgsz` 를 체크포인트에서 복원해 `model.overrides` 에 넣습니다** |
> | ③ `predict()` | `args = {**self.overrides, **custom, **kwargs}` — `custom` 에 `imgsz` 없음 |
> | ④ sahi 0.12.5 | `image_size` 가 `None` 이면 `kwargs` 에 `imgsz` 를 **안 넣습니다** |
>
> → 우리가 `image_size` 를 안 넘겨도 **체크포인트의 1280 이 그대로 적용됩니다.**
> 슬라이스 1280 은 리사이즈 없이 1280 으로 들어갑니다.
>
> **왜 640 이라고 적었었나**: 진단 코드가 `get_cfg()` 를 읽었습니다. 그건 글로벌 `DEFAULT_CFG` 이지
> 이 모델이 쓰는 값이 아닙니다. `iou 0.7` · `max_det 300` 은 `include` 집합에 없어서 그 방법이
> **맞지만**, `imgsz` 는 복원되는 4개 키 중 하나라 그 방법이 안 통합니다.
> `fixtures/env.json` 에 `ultralytics_default_imgsz` 키가 아예 없다는 것도 방증입니다 —
> **한 번도 측정된 적이 없는 값이었습니다.**
>
> **성능 수치·골든 픽스처는 무효가 아닙니다.** 실제로 돌아간 경로에서 잰 값이고 라벨만 틀렸습니다.
> `image_size=1280` 을 명시해도 값이 같으므로 결과는 달라지지 않습니다.
>
> ### ✅ 런타임 확인 완료 (백엔드, 2026-08-07)
>
> 소스 확인과 별개로 배포 측에서 실제로 돌려 확인했습니다.
>
> | 경로 | 결과 |
> | --- | --- |
> | `image_size` 미지정 (= 배포 경로) | **골든 픽스처가 정확히 재현됨** |
> | `imgsz=640` 강제 | **같은 골든 이미지에서 검출 0건** |
>
> **골든 픽스처가 640 에서 0건이 된다는 것 자체가 증명입니다** — 픽스처는 배포 경로에서 만든
> 스냅샷이므로, 640 이 정본이었다면 픽스처가 640 에서 재현돼야 합니다. 그 반대가 나왔습니다.
> **`imgsz = 1280` 확정.** 640 으로 되돌리는 변경은 검출을 전부 죽입니다.

> ### ⚠️ ROI 크기는 고정이 아닙니다
>
> CT 는 축 단면이라 축마다 이미지가 담는 축 조합이 바뀝니다. 셀 101 기준 y축 단면은 `562 x 4000`,
> z축 단면은 `562 x 1152` 입니다. 셀마다도 다릅니다(폭 중앙값 411). **크기를 하드코딩하지 마세요** —
> 좌표계는 매 JSON 의 `coordinate_space.roi_size` 에 실려 나갑니다.

> ### 채널 순서
>
> 우리 경로는 sahi 에 **파일 경로**를 넘기고, sahi 가 RGB 로 읽어 내부에서 BGR 로 뒤집어 ultralytics 에
> 넘깁니다. numpy 배열로 직접 넘기려면 **RGB 여야 합니다** — `cv2.imread` 결과(BGR)를 그대로 넣으면
> 채널이 뒤집혀 에러 없이 성능만 떨어집니다. 배열로 넘길 거면 §7 골든 픽스처로 먼저 확인하세요.

---

## 4. 추론 파라미터

```yaml
model:
  weight: ct_samedist_CHAMPION_ep6.pt
  task: segment
  classes: { 0: porosity }
  device: cuda:0

sahi:
  enabled: true                     # CT 는 ON. RGB 는 OFF (계약서)
  slice_height: 1280                # = 학습 TILE. 바꾸지 말 것
  slice_width: 1280
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  postprocess_type: NMS             # 명시 필수 — 기본값은 conf 에 따라 바뀜
  postprocess_match_metric: IOU
  postprocess_match_threshold: 0.5

detection:
  confidence_threshold: 0.05        # 배포 운영점 (samedist ep6 전용)
```

### 코드가 지정하는 값 / 라이브러리 기본값

**코드가 명시적으로 지정하는 건 위 yaml 뿐입니다.** 나머지는 전부 라이브러리 기본값이고,
**버전이 바뀌면 조용히 값이 바뀝니다.** 서빙에서는 기본값에 기대지 말고 명시적으로 넘겨 고정하세요.

| 층 | 파라미터 | 값 | 출처 |
| --- | --- | --- | --- |
| ① 슬라이스 내부 NMS | `iou` | `0.7` | 미지정 → ultralytics 8.4.115 기본값 (`get_cfg` 실측) |
| ① 슬라이스 내부 NMS | `max_det` | `300` | 미지정 → ultralytics 8.4.115 기본값 (`get_cfg` 실측) |
| ① 슬라이스 내부 | `agnostic_nms` | `false` | 미지정 → ultralytics 기본값. 단일 클래스라 영향 없음 |
| ① 슬라이스 내부 | `half` | `false` (fp32) | 미지정 → ultralytics 기본값 **추정**. §9 |
| ① 슬라이스 내부 | `imgsz` | **`1280`** | 미지정이나 **체크포인트에서 복원**됨 (기본값 640 아님). §3 |
| ② 슬라이스 간 병합 | `postprocess_type` / `match_metric` | `NMS` / `IOU` | **명시 지정** |
| ② 슬라이스 간 병합 | `postprocess_match_threshold` | `0.5` | **명시 지정** |
| ② 슬라이스 간 병합 | `postprocess_class_agnostic` | `false` | 미지정 → sahi 기본값. 단일 클래스라 영향 없음 |
| ② SAHI | `perform_standard_pred` | `true` | 미지정 → sahi 기본값. 아래 주의 |
| ② SAHI | `batch_size` | `1` | 미지정 → sahi 기본값 |
| ③ 평가 전용 | `IOU_HIT` | `0.1` | 채점용 localization 판정. **추론 경로와 무관** |

③을 NMS 값으로 오해하지 마세요. 평가 스크립트에만 있는 값입니다.

> ### ⚠️ sahi 는 conf 가 낮으면 병합 방식을 조용히 바꿉니다
>
> 0.12.5 에서 배포 운영점 `conf 0.05` 로 돌리면 기본값 `GREEDYNMM`/`IOS` 대신 `NMS`/`IOU` 로 자동
> 전환되고 경고만 찍습니다 (`Switching postprocess type/metric to NMS/IOU since model confidence
> threshold is low`). 그래서 코드에 `NMS`/`IOU` 를 **명시**했습니다 — 값은 그대로지만 conf 를 올려도
> 병합 방식이 안 바뀝니다.
>
> 명시했다고 안전한 게 아닙니다. `force_postprocess_type=True` 를 주지 않으면 **명시값도 자동전환
> 대상**입니다. 지금은 명시값과 전환 결과가 같아서 무해할 뿐이고, `GREEDYNMM` 을 쓰려면 그 플래그가 필요합니다.
>
> **conf 를 0.05 보다 올려 재평가하면 자동 전환이 안 걸려 배포와 다른 방식으로 병합됩니다.**
> 평가에서 conf 를 올려 잰 수치를 배포 수치와 직접 비교할 때 이 점을 감안하세요.

> ### `perform_standard_pred` 가 켜져 있습니다
>
> sahi 기본값이 `true` 라, 슬라이스 추론 외에 **원본 전체를 한 번 더 추론하고 병합**합니다
> (이때도 `imgsz` 는 1280 이라 긴 변이 1280 에 맞춰 리사이즈됩니다 — 원본이 `562x4000` 이면 큰 축소입니다).
> 코드가 끄지 않았으므로 배포도 이 상태이고, §6 수치도 이 상태에서 잰 값입니다. 끄면 결과가 달라집니다.

---

## 5. 출력 매핑

| 항목 | 값 |
| --- | --- |
| `model_classes` | `{"0": "porosity"}` — **단일 클래스** |
| `class_map` | `{"porosity": "MICRO_DEFECT"}` |
| `bbox_format` | `xyxy` |
| 계약용 `{x,y,width,height}` | 출력 JSON 의 **`bbox_xywh` 필드가 그대로 대응**. 변환 불필요 |
| `coordinate_reference` | `roi_image` |
| origin / unit | `top-left` / `pixel` |
| `verdict` | `num_detections > 0` → `REJECT`, 아니면 `PASS` |

`detections[].class` 에는 정수 id 가 아니라 문자열 `porosity` 가 들어갑니다.
`data.yaml` 의 `names: ['porosity']` 가 유일한 출처이고, 새 결함 유형을 임의로 만들지 않습니다(계약서 Core §6.5).
CT 가 낼 수 있는 계약 enum 은 `MICRO_DEFECT` 하나뿐입니다.

### 좌표계

납품 이미지 자체가 ROI 입니다. `bbox_xyxy` 는 **그 이미지의 픽셀 좌표**이고 origin 은 좌상단입니다.
**좌표 변환·오프셋이 필요 없습니다.**

CT 한 장은 단면이라 bbox 가 2축만 담습니다. 나머지 1축은 `cell.axis` + `cell.slice_index` +
`volume.axis_mapping` 이 채웁니다. 자세한 건 스키마 샘플의 `coordinate_space` · `volume` 블록과
`battery_infer_to_json.ipynb` §4(자체검증 게이트)를 보세요.

물리 단위(mm)는 제공하지 않습니다 — 원본에 voxel spacing 메타데이터가 없어 픽셀/복셀 단위까지만 나갑니다.

---

## 6. 운영점 (conf)

| conf | 성격 | 실측 |
| --- | --- | --- |
| 0.02 | 최대 recall | R 0.990 · 재검부하 11.8% |
| **0.05** | **배포 기본값** | **R 0.977 · loc 86.7% · 재검부하 6.6%** |
| 0.10 | F1 최대 | F1 0.921 (모델 비교용, 배포용 아님) |

불량 게이트라 **recall 우선**입니다. 놓친 불량이 오탐보다 훨씬 비쌉니다 — 오탐은 사람 재검으로 회수되지만
미검출은 그대로 나갑니다.

> **`conf` 는 가중치와 짝입니다.** `0.05` 는 `ct_samedist_CHAMPION_ep6.pt` 전용입니다. 다른 가중치면
> `0.25` 이고, 위 수치는 성립하지 않습니다. `ai-infer` 는 기동 시
> `model_card.inference.conf_is_paired_with_weight` 와 실제 로드한 가중치 파일명을 대조하세요.

측정 조건: **samedist ep6 + slice 1280 / overlap 0.2 + imgsz 1280 + `perform_standard_pred` on.**
넷 중 하나만 바꿔도 위 실측값은 무효입니다.

평가 split 은 **동분포(known-type)** 입니다 — 47셀 전부 train·val 양쪽에 있고 이미지 단위 stratified.
새 셀 타입에서는 눈에 띄게 떨어집니다(§9).

---

## 7. 골든 픽스처 (회귀 테스트)

배포·리팩터링·라이브러리 업그레이드 후 **같은 입력에서 같은 출력이 나오는지** 확인하는 고정 표본입니다.
전처리가 어긋나도 서버는 200 을 반환하고 성능만 조용히 나빠지므로, 이게 유일한 자동 방어선입니다.

```
fixtures/
  golden_ct.jpg      # 입력 이미지 (결함 있는 것)
  golden_ct.json     # 위 설정으로 낸 기대 출력
  golden_ct.sha256   # 입력 해시 (파일이 바뀌면 비교 자체가 무의미)
  env.json           # 생성 환경 + ultralytics 실측 기본값
```

### 현재 픽스처 (2026-08-03)

| | |
| --- | --- |
| 장수 | **1장** (요청은 20~30장 — §11) |
| 이미지 | `CT__CT_cell_pouch_101_z_119__b82e0912.jpg` (셀 101, z축 119번, **562 x 1152**) |
| sha256 | `2f749661c58cee23a6a8cecf5ea195647e86c33411e34649e37035dbd6c2d97f` |
| 기대 출력 | `verdict REJECT` · 검출 1건 · `porosity` · conf **0.2705** · bbox `[269, 620, 274, 912]` |
| 임계 여유 | **+0.2205** (conf 0.05 기준) |
| 생성 환경 | Colab / Tesla T4 · ultralytics 8.4.115 · sahi 0.12.5 · torch 2.11.0+cu128 |
| 검증 | **§6 통과 확인 (2026-08-03)** — 회귀 테스트가 실제로 도는 것까지 확인됨 |

기준 이미지는 **"임계에서 가장 멀리 떨어진 검출"이 있는 장**으로 고릅니다. conf 가 임계(0.05)에
붙어 있는 케이스를 굳히면 라이브러리·GPU 가 조금만 달라져도 그 검출이 사라져 `verdict` 가 뒤집힙니다.
지금 픽스처는 임계에서 0.22 떨어져 있어 안전합니다.

### 통과 기준

| 항목 | 허용 오차 |
| --- | --- |
| `verdict` · `num_detections` | **완전 일치** |
| `detections[].class` | **완전 일치** |
| `bbox_xyxy` | ±1.0 px |
| `confidence` | ±0.01 |
| `segmentation` | 폴리곤 점 개수가 달라질 수 있어 **면적 ±5%** 로만 비교 |

`verdict` 나 검출 개수가 달라지면 배포를 멈추세요. bbox 가 1px 넘게 움직이면 대개 라이브러리 버전이 바뀐 것입니다.

### 생성 · 검증

`battery_infer_to_json.ipynb` **§5** 가 생성, **§6** 이 회귀 검증입니다.
픽스처는 **§2 가 실제로 낸 출력을 그대로 굳힙니다** — 픽스처용으로 따로 추론하면 배포 경로와 다른
코드를 검증하게 되므로 일부러 재추론하지 않습니다.
라이브러리 업그레이드·리팩터링 후에는 **§1 → §2 → §6** 순으로 돌리면 됩니다.

---

## 8. 학습 재현

| 항목 | 값 |
| --- | --- |
| 데이터 | v4.1, `nobig` 정책 (`porosity_bbox_max_ratio ≥ 0.25` 인 대형 오라벨 이미지 제외) |
| split | **동분포(known-type)** — 셀 그룹분할이 아니라 이미지 단위 stratified, 47셀 전부 train·val 양쪽, 셀당 15% val |
| seed | 42 |
| 타일 | `TILE = ceil(max(1280, Wmax)/32)*32` → 실측 1280. 가로 1칸 · 세로 스트립, 폭 네이티브 보존 |
| 타일 라벨 | shapely 교집합 클립 — 스트립에 걸친 결함을 조각 라벨로 분배(중심 배정하면 걸친 스트립이 배경 오라벨이 됨) |
| 배경 타일 | `BG_KEEP 0.08`, `N_BG_TILES 1` |
| imgsz | **`= TILE` (1280). 학습은 다운스케일 없음** — 추론과 다릅니다(§3) |
| batch | 12 (OOM 시 8) |
| optimizer | AdamW, `lr0 0.0005`, `cos_lr`, `amp=True` |
| epochs / patience | 18 / 15 (채택은 6) |
| 증강 | `hsv_h 0.0, hsv_s 0.0, hsv_v 0.1, degrees 10, flipud 0.5, translate 0.2, scale 0.1, copy_paste 0.0` |
| seg 옵션 | `mask_ratio 2`, `overlap_mask False` |
| 학습 중 val | **`val=False`** — CT 는 mask mAP50 이 함정이라 학습 중 val 을 쓰지 않고, conf 스윕 F1 으로 판정 |
| 학습 환경 | RunPod A40 / Colab |

노트북: `battery_ct_tiled_v41_samedist.ipynb` (§1 config → §2 타일 빌드 → §3 학습 → §4 평가 → §6 챔피언 백업)

---

## 9. 한계 · 알아야 할 것

- **새 셀 타입에서는 성능이 떨어집니다.** §6 수치는 학습에서 본 것과 같은 타입(동분포) 기준입니다.
  새 타입에서는 눈에 띄게 하락하며 재보정/파인튜닝이 필요합니다. 배포 시나리오를 "같은 라인·같은 타입
  재검사"로 볼지 "새 타입 유입"으로 볼지에 따라 기대치가 달라집니다.
- **mask mAP50 으로 이 모델을 평가하지 마세요.** 게이트가 하는 일은 마스크를 정확히 그리는 게 아니라
  불량 셀을 플래그하는 것이고, mAP 기준으로 고르면 실제로 더 나은 모델을 버리게 됩니다. 판정은
  image-level PASS/REJECT F1 과 localization-aware F1 로 합니다.
- **슬라이스 · conf · 가중치 · imgsz 는 한 세트입니다.** 하나만 바꾸면 §6 실측값은 무효입니다.
- **`half` 는 실측이 아니라 추정입니다.** ultralytics 기본 fp32 로 알고 있으나 `env.json` 에 안 찍혀
  있습니다. T4/EC2 에서 fp16 으로 배포하려면 **골든 픽스처를 fp16 에서 다시 만들어야 합니다**
  (confidence 가 ±0.01 허용오차를 넘길 수 있음).
- ~~실효 imgsz 640 은 개선 여지~~ **폐기 (0806)**. 이미 1280 으로 돌고 있었습니다(§3).
  `image_size=1280` 을 명시해도 값이 같아 아무것도 안 바뀝니다. **이 실험에 시간 쓰지 마세요.**
  부수 효과로 타일링의 논거("폭을 네이티브로 보존한다")가 추론에서도 깨지지 않고 유지됩니다.
- **`imgsz` 는 소스 + 런타임 양쪽으로 확인됐습니다**(§3). 다만 `fixtures/env.json` 에는 아직
  값이 안 박혀 있습니다 — `battery_infer_to_json.ipynb §5` 를 다시 돌리면 `effective_imgsz` 로
  기록되고, `SLICE` 와 다르면 assert 로 멈춥니다.
- `volume.slice_scale_px` 는 슬라이스가 ROI 를 균등히 덮는다는 가정의 근사값입니다.

---

## 10. `confidence` 산식 (요청 §7 답변)

계약 §12.1 은 `PASS`(후보 있음) 의 최상위 `confidence` 를 `1 - (임계값 미만 최고 결함 점수)` 로 정의하는데,
CT 는 `conf=0.05` 로 돌리므로 0.05 미만 후보가 애초에 반환되지 않습니다.

**제안: 모델 conf 를 바닥값(예 0.01)으로 낮춰 돌리고, 0.05 판정은 애플리케이션 코드에서 합니다.**
그러면 0.05 미만 최고 점수를 그대로 얻을 수 있습니다.

이게 성립하는 이유는 **병합이 `NMS`/`IOU` 로 고정돼 있기 때문**입니다. NMS 는 점수가 높은 박스가 낮은
박스를 지우는 방향으로만 작동하므로, 낮은 점수 후보를 더 넣어도 이미 살아남은 0.05 이상 박스는 영향을
받지 않습니다. `max_det 300` 도 점수 상위부터 채우므로 (0.05 이상 박스가 300개를 넘지 않는 한) 마찬가지입니다.
따라서 **conf 0.01 로 돌린 뒤 0.05 로 필터링한 결과 = conf 0.05 로 돌린 결과** 여야 합니다.

⚠️ 두 가지 단서:

1. **`GREEDYNMM`/`NMM` 에서는 성립하지 않습니다.** 그쪽은 박스를 지우는 게 아니라 **병합**하므로,
   저점수 후보가 늘면 살아남는 박스의 모양 자체가 바뀝니다. §4 의 명시 지정을 풀면 이 제안도 무효입니다.
2. **위는 NMS 의 성질에서 나온 추론이지 실측이 아닙니다.** 검증은 쌉니다 — `CONF = 0.01` 로 §2 를 돌리고
   0.05 이상만 남긴 뒤 §6 을 태워서 골든 픽스처와 일치하면 확정입니다. 통과하면 이 변경은 공짜입니다.

비용은 병합에 들어가는 박스 수가 늘어나는 만큼의 지연 증가뿐입니다.
값이 비싸다고 판단되면 대안은 계약 산식을 `PASS` 일 때 `1.0` 고정으로 단순화하는 쪽입니다.

---

## 11. 남은 갭

| 항목 | 상태 | 채우는 법 |
| --- | --- | --- |
| `weight_sha256` | 미기입 | Drive 의 사이드카 `ct_samedist_CHAMPION_ep6.json` 에 있음 |
| `trained_at` | 미기입 | 학습 run 폴더 타임스탬프 확인 |
| `half` 실측 | 추정 | `battery_infer_to_json.ipynb` §5 의 `get_cfg()` 캡처에 포함시켜 재실행 |
| 골든 픽스처 장수 | 1 / 요청 20~30 | §2 가 이미 20장(REJECT 12 / PASS 8)을 돌립니다. §5 가 그중 1장만 굳히는 것이라, 전부 굳히도록 §5 를 늘리면 됩니다 |

---

## 12. 관련 파일

| 파일 | 역할 |
| --- | --- |
| `ct.model_card.json` | 이 문서의 기계 판독 형태 (S3 배포본) |
| `ct_output_schema.sample.json` | 출력 JSON 계약 (정본) |
| `requirements.lock.txt` | 환경 고정 |
| `fixtures/` | 골든 픽스처 |
| `battery_infer_to_json.ipynb` | 추론 → JSON + 오버레이 + 좌표계 자체검증 + 픽스처 생성/검증 |
| `battery_ct_tiled_v41_samedist.ipynb` | 학습 파이프라인 · 챔피언 백업 |
| `battery_ct_tiled_v41_samedist_score_check.ipynb` | conf 스윕 점수 검증 |
