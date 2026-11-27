---
title: "Cross-channel negation: kết quả theo giả thuyết (B200, cập nhật 2026-11-27 04:45 UTC)"
author: "Pico, cho buổi họp của Vy"
geometry: margin=1.8cm
fontsize: 10pt
---

**Tóm tắt nghiên cứu.** Các omni-model (nhận image + audio + text) được cho một lệnh cấm như "Do not mention the dog."
qua ba kênh: gõ thành một khối text riêng (**T**), nói bằng giọng (**A**), hoặc viết trong một ô trong ảnh (**V**).
Câu hỏi luôn ở dạng text. **Violation rate** là tỉ lệ câu trả lời vẫn nhắc tới thành phần bị cấm. Nó chỉ tính trên
các câu mà chính mô hình đó nhắc tới thành phần ấy khi không có lệnh (**C0**).

Năm mô hình: Qwen2.5-Omni-7B, Qwen2.5-Omni-3B, MiniCPM-o 2.6, Phi-4-multimodal, InternOmni. Bộ đầy đủ có 4.338 câu
hỏi trên 2.627 ảnh (các unit asr_ok). Pilot là một danh sách cố định 1.000 câu, mỗi branch 200 câu. CI 95% được
tính bằng bootstrap theo ảnh. Kiểm định cặp dùng McNemar exact.

**Lưu ý chung.**

1. LLM judge (Qwen3-8B) **chưa được người kiểm tra trên câu trả lời của B200**. Mới chỉ có 49 câu trên server 1:
   đồng ý 93,9%, κ 0,83.
2. Mọi phần trừ H1 đều chạy trên **pilot 1.000 câu**, nên không kết luận gì theo branch hay theo split từ pilot.
3. Chỉ so sánh các kênh **trong cùng một mô hình**.

# H1: cùng một lệnh cấm, khi được gõ, được nói, hay được viết trong ảnh, có được tuân theo như nhau không? (bộ đầy đủ)

**Kết quả H1:** violation rate (%) trên các câu C0-mentioned. Cột T − A là hiệu theo cặp, kèm CI 95%.

| model | n | T | A | V | T − A [95% CI] | recall C0 / T / A / V |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 3.535 | 17,3 | 13,5 | 89,0 | +3,8 [+2,6, +5,0] | 80,4 / 84,1 / 83,8 / 79,5 |
| Qwen2.5-Omni-3B | 3.558 | 65,4 | 50,3 | 90,6 | +15,1 [+13,5, +16,8] | 81,4 / 81,8 / 82,7 / 78,4 |
| MiniCPM-o 2.6 | 3.441 | 48,1 | 30,1 | 93,9 | +17,9 [+16,2, +19,7] | 80,3 / 84,6 / 84,1 / 79,1 |
| Phi-4-MM | 3.478 | 27,6 | 37,4 | 87,2 | −9,8 [−11,9, −7,7] | 80,5 / 63,8 / 68,1 / 78,0 |
| InternOmni | 3.483 | 49,2 | 80,8 | 94,7 | −31,6 [−33,7, −29,7] | 71,2 / 73,3 / 25,3 / 72,7 |

\small Nguồn: experiments/h1_v2/full_analysis_5models_asrok.md. Kiểm tra trên mọi unit: full_analysis_5models_allunits.md.
Bỏ các dòng judge lỗi: analysis_all_asrok_rm.md. Recall là tỉ lệ các thành phần được phép mà mô hình vẫn nêu.\normalsize

![Violation rate theo kênh, 5 mô hình (bộ đầy đủ, asr_ok).](../experiments/figures/full_h1v2_violation_5models_asrok.png){width=68%}

**=> Rút ra findings là:**

- Lệnh gõ và lệnh nói được tuân theo khác nhau ở **cả 5 mô hình** (mọi CI đều không chứa 0), nhưng **chiều khác nhau
  theo mô hình**. Vì vậy không có kênh nào "yếu hơn" nói chung.
- Lệnh viết trong ảnh (V) bị vi phạm 87–95% ở cả 5 mô hình.
- Recall của InternOmni tụt còn 25% khi lệnh được nói. Mô hình có vẻ coi câu nói là câu hỏi, nên violation rate của nó
  ở kênh A không phản ánh sạch việc "lờ lệnh".
- Kết quả vững: T − A lệch ≤ 0,2 điểm khi tính trên mọi unit, ≤ 0,3 điểm khi bỏ các dòng judge lỗi, và giữ nguyên
  chiều khi không tính kiểu trả lời "X không có".

# Controls (C0_box / OCR_V): mô hình có đọc được lệnh viết trong ảnh không, và một ô trống có làm thay đổi câu trả lời không? (pilot)

**Kết quả controls:**

| model | OCR_V exact copy % (n=1.000) | C0 → C0_box, % nêu target | V − C0_box (điểm) [95% CI], trên C0-mentioned |
|---|---|---|---|
| Qwen2.5-Omni-7B | 99,8 | 84,4 → 83,9 (p=0,60) | −6,0 [−8,0, −4,0] |
| Qwen2.5-Omni-3B | 93,6 | 84,0 → 81,7 (p=0,007) | −2,1 [−3,9, −0,4] |
| MiniCPM-o 2.6 | 99,7 | 81,9 → 83,0 (p=0,13) | −3,2 [−4,9, −1,6] |
| Phi-4-MM | 98,1 | 81,2 → 80,2 (p=0,28) | −5,2 [−7,4, −3,1] |
| InternOmni | 99,0 | 82,6 → 82,6 (p=1) | −0,7 [−2,2, +0,8] |

\small Nguồn: experiments/h1_v2/pilot1000_comp_ctrl_5models.md. V − C0_box được tính lại từ dữ liệu thô trong
audit/number_audit.md: điểm ước lượng trùng khớp, CI lệch ≤ 0,2.\normalsize

**=> Rút ra findings là:**

- Mọi mô hình đều đọc được lệnh viết trong ảnh (chép đúng 94–100%), và ô trống hầu như không làm thay đổi câu trả lời.
- So với ô trống, viết lệnh vào ảnh chỉ làm giảm việc nhắc target 0–6 điểm. Vậy lỗi ở kênh V **không phải do không đọc
  được**: chữ được đọc, nhưng phần lớn không được coi là lệnh.

# H3-B: mô hình có hiểu lệnh cấm ở từng kênh không, tách riêng khỏi việc có làm theo hay không? (COMP v1 xong; COMP v2 đang chạy)

**Kết quả H3-B:**

- **COMP v1 không hợp lệ.** Câu hỏi được đặt sau một lệnh còn đang có hiệu lực, nên mô hình áp lệnh vào luôn câu trả
  lời trắc nghiệm. Ví dụ Qwen-7B ở T đúng 21,2% và MiniCPM-o ở T đúng 8,8%, đều thấp hơn mức ngẫu nhiên 33%
  (analysis_comp_pilot1000.md).
- **COMP v2** trích dẫn lệnh trước rồi mới hỏi về nó. Bảng dưới là pilot, mới xong 2/5 mô hình.

| model | hiểu lệnh % T / A / V [95% CI] | lặp nguyên văn exact % T / A / V | V violation khi đã hiểu |
|---|---|---|---|
| Qwen2.5-Omni-7B | 100,0 / 99,9 [99,7, 100] / 100,0 | 99,3 / 97,1 / 99,8 | 90,4 [88,4, 92,4] (n=821) |
| Qwen2.5-Omni-3B | 99,7 [99,3, 100] / 99,8 [99,5, 100] / 100,0 | 92,2 / 95,6 / 93,6 | 92,8 [90,9, 94,5] (n=805) |
| MiniCPM-o, Phi-4, InternOmni | đang chạy | đang chạy | đang chạy |

\small Nguồn: experiments/h1_v2/analysis_comp2_pilot1000.md (một phần; chạy lại khi job-comp2 xong, ETA khoảng 1–3 giờ).
Lưu ý: khi được bảo lặp lại câu gõ, MiniCPM-o lặp câu đó thành vòng, nên điểm "exact" bị thấp giả. Phần này sẽ được báo
kèm một phép kiểm tra phần đầu câu.\normalsize

**=> Rút ra findings là:**

- Cả hai mô hình Qwen hiểu lệnh cấm ở **cả ba kênh** (khoảng 100%), kể cả V.
- Ở 90–93% các câu đã hiểu, chúng vẫn vi phạm lệnh **viết trong ảnh**. Hiểu lệnh không phải là điểm nghẽn của V.
  Cần đợi 3 mô hình còn lại trước khi khái quát (chưa chắc).

# H4: khoảng cách giữa các kênh có riêng cho phủ định không, khi so với một lệnh tích cực? (pilot)

Delta_c = compliance(lệnh tích cực) − compliance(lệnh phủ định) ở kênh c. Lệnh tích cực có hai dạng: affirmative
"Mention only Y." và exceptive "Mention everything except X.".

**Kết quả H4:** đơn vị là điểm, kèm CI 95%. Số unit: affirmative 647–669, exceptive 801–833.

| model | Delta_T (aff) | Delta_A (aff) | Delta_A − Delta_T (aff) | Delta_A − Delta_T (exc) | Delta_V (aff) |
|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | −19,6 | −23,2 | −3,6 [−8,2, +0,9] | −0,1 [−4,1, +4,2] | +4,3 |
| Qwen2.5-Omni-3B | +19,1 | +14,9 | −4,2 [−9,0, +0,6] *fragile* | −19,3 [−24,1, −14,7] | −1,1 |
| MiniCPM-o 2.6 | −8,9 | +2,9 | +11,9 [+6,6, +17,6] | +5,7 [+0,6, +10,9] *fragile* | +0,6 |
| Phi-4-MM | +9,3 | +21,2 | +11,9 [+7,0, +17,0] | +8,1 [+3,2, +13,1] | −1,9 |
| InternOmni | +19,8 | +56,1 | +36,3 [+30,9, +41,8] | +0,4 [−4,8, +5,5] | +0,5 |

\small Nguồn: experiments/h1_v2/pilot1000_forms_5models_asrok.md. "Fragile" nghĩa là CI vượt qua 0 khi bỏ các unit judge
lỗi (analysis_pilot1000_forms_asrok_rm.md).\normalsize

![H4: compliance theo dạng lệnh và kênh (pilot).](../experiments/figures/pilot1000_h4_forms_5models_asrok.png){width=68%}

**=> Rút ra findings là:**

- Khi so với affirmative, phủ định qua giọng nói bị phạt nặng hơn qua text ở 3 mô hình: MiniCPM-o, Phi-4, InternOmni.
  Ở hai mô hình Qwen, CI chứa 0.
- So với exceptive (có nêu X nhưng không có "not"), hiệu ứng này **nhỏ lại hoặc đổi chiều**. InternOmni còn +0,4, vì kết
  quả affirmative của nó bị nhiễu bởi việc "trả lời câu nói". Qwen-3B thành −19,3. Vậy việc gap có riêng cho phủ định hay
  không **tùy theo mô hình**.
- Lệnh viết trong ảnh bị lờ ở **mọi** dạng lệnh (V violation 86–97%), nên Delta_V không nói lên điều gì về phủ định.
- Đây chỉ là pilot. Bản mở rộng H4 lên bộ đầy đủ cho hai mô hình Qwen đang chạy (xem bên dưới).

# H2: khi câu hỏi và lệnh cấm đi qua hai kênh khác nhau, lệnh cấm có được tuân theo như nhau bất kể nó đi kênh nào không? (pilot)

Ký hiệu: q là kênh của câu hỏi, i là kênh của lệnh. Lưu ý: đây **không phải** H2 trong tài liệu Idea (gõ "Mention X"
cộng nói "Do not mention X"). Phiên bản đó chưa chạy.

**Kết quả H2:** violation rate (%). Hiệu ứng chiều = qT_iA − qA_iT, kèm CI 95%; n = 800–832.

| model | qT_iA | qA_iT | qT_iA − qA_iT | lệnh trong ảnh (qT_iV / qA_iV) | câu hỏi trong ảnh (qV_iT / qV_iA) |
|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 13,2 | 17,5 | −4,3 [−7,0, −1,6] | 90,1 / 90,0 | 19,7 / 17,4 |
| Qwen2.5-Omni-3B | 49,2 | 67,8 | −18,6 [−22,1, −15,1] | 92,5 / 90,8 | 44,5 / 41,4 |
| MiniCPM-o 2.6 | 30,4 | 54,3 | −23,9 [−27,7, −20,3] | 94,9 / 95,3 | 54,8 / 38,7 |
| Phi-4-MM | 42,6 | 32,6 | +10,0 [+6,0, +13,9] | 90,8 / 88,6 | 30,6 / 43,0 |
| InternOmni | 80,1 | 42,9 | +37,2 [+32,9, +41,4] | 95,8 / 84,9 | 29,9 / 68,9 |

\small Nguồn: experiments/h1_v2/analysis_h2_pilot1000_asrok.md.\normalsize

**=> Rút ra findings là:**

- Giữa gõ và nói, chiều của lệnh có ảnh hưởng ở cả 5 mô hình. Kênh làm tốt hơn trùng với kênh làm tốt hơn trong H1 của
  chính mô hình đó.
- Lệnh viết trong ảnh bị vi phạm 85–96% dù câu hỏi đi qua kênh nào.

# H3-A: cùng các bố cục đó nhưng không có phủ định ("Mention only Y."), kênh nào thắng? (pilot)

Mục đích là tách modality dominance nói chung ra khỏi hiệu ứng riêng của phủ định. Branch open existence không có dạng
affirmative nên bị bỏ, n = 646–668.

**Kết quả H3-A:**

| model | H3-A qT_iA − qA_iT | H2 − H3-A (T–A) | H2 − H3-A, bỏ unit judge lỗi | lệnh tích cực khi câu hỏi trong ảnh (qV_iT / qV_iA) |
|---|---|---|---|---|
| Qwen2.5-Omni-7B | +2,7 [−1,0, +6,4] | −7,9 [−12,7, −3,3] | −8,6 [−13,9, −2,9] | 19,5 / 18,7 |
| Qwen2.5-Omni-3B | +1,4 [−1,8, +4,5] | −15,4 [−20,2, −10,4] | −17,6 [−23,3, −11,7] | 18,7 / 13,7 |
| MiniCPM-o 2.6 | −11,4 [−15,1, −7,7] | −13,9 [−19,5, −8,3] | −14,5 [−20,4, −8,7] | 19,0 / 20,1 |
| Phi-4-MM | +5,9 [+2,5, +9,1] | +5,9 [+0,2, +11,5] *sát ngưỡng* | +6,6 [+0,4, +12,6] | 15,9 / 16,6 |
| InternOmni | +5,9 [+2,1, +9,7] | +32,5 [+26,9, +38,2] | +32,9 [+26,5, +38,8] | 8,8 / 10,3 |

\small Nguồn: experiments/h1_v2/analysis_h2_pilot1000_asrok.md và analysis_h2_pilot1000_asrok_rm.md.\normalsize

**=> Rút ra findings là:**

- Hiệu ứng chiều gõ/nói **lớn hơn khi có phủ định so với khi không có, ở cả 5 mô hình**: CI của H2 − H3-A không chứa 0,
  kể cả khi bỏ các unit judge lỗi. Riêng Phi-4 sát ngưỡng. Đây là pattern riêng cho phủ định nhất quán nhất cho tới
  giờ, và chiều của nó theo đúng gap H1 của từng mô hình.
- Kết quả của InternOmni bị nhiễu vì mô hình coi câu nói là câu hỏi: recall chỉ 10–27% khi lệnh cấm được nói.
- Khi câu hỏi nằm trong ảnh, lệnh tích cực được tuân theo tốt (vi phạm 9–20%). Lệnh cấm thì kém hơn nhiều ở 4/5 mô
  hình (vi phạm 30–69%).

# H3-D: target càng lớn thì càng dễ bị vi phạm? Kèm bảng nguồn × split (bộ đầy đủ, chỉ CPU, đã xong)

**Kết quả H3-D:** violation (T) ở tertile target lớn trừ tertile target nhỏ, kèm CI 95%.

| model | T | A | V |
|---|---|---|---|
| Qwen2.5-Omni-7B | +9,8 [+6,5, +13,1] | +7,2 [+4,4, +10,2] | +1,8 [−0,9, +4,6] |
| Qwen2.5-Omni-3B | +6,6 [+2,4, +10,7] | +13,3 [+9,1, +17,6] | −4,0 [−6,4, −1,7] |
| MiniCPM-o 2.6 | +5,6 [+0,9, +10,0] | +3,8 [−0,1, +7,8] | +0,1 [−2,1, +2,4] |
| Phi-4-MM | +4,3 [+0,4, +8,2] | +9,4 [+5,1, +13,7] | +1,0 [−2,0, +3,8] |
| InternOmni | −2,6 [−7,4, +2,1] | −6,6 [−9,7, −3,3] | −1,9 [−3,7, −0,0] |

\small Nguồn: experiments/h1_v2/h3d_area_b200.md, split_by_source_b200.md, dataset_stats_b200_full.md.\normalsize

**=> Rút ra findings là:**

- Ở 4/5 mô hình, target lớn bị vi phạm nhiều hơn ở T và A. InternOmni đi ngược lại.
- Đây chỉ là quan sát và bị lẫn với branch: hiệu ứng kích thước rõ nhất trong branch attribute. B1 chỉ giữ các vật
  chiếm ≥ 3% ảnh.
- CLEVR chỉ có val, còn COCO gần như toàn train, nên **không thể kết luận train vs val**.

# NEU control: mô hình có kém ở việc làm theo BẤT KỲ lệnh nào trong một kênh, hay chỉ kém ở phủ định?

Lệnh trung tính là "Include the word W in your answer.", đặt cùng kênh, cùng vị trí, cùng giọng và cùng kiểu ô với lệnh
cấm.

**Kết quả: Đang chạy.** Qwen7B/3B đã sinh xong. MiniCPM-o 3.581/6.000, InternOmni 3.041/6.000, Phi-4 721/6.000 (chậm
nhất, khoảng 8 giờ ở tốc độ GPU dùng chung). Sau đó judge chấm từng mô hình một, khoảng 1,2 giờ mỗi mô hình.

**Findings mong đợi:**

- Nếu (NEU follow − NEG follow) gần như nhau ở T, A, V, thì gap của H1 là do khả năng làm theo lệnh nói chung.
- Nếu nó khác nhau theo kênh, thì gap đó riêng cho phủ định.
- Bảng 2×2 {V, V + "The image contains an instruction for you."} × {NEG, NEU} sẽ cho biết việc báo cho mô hình rằng
  trong ảnh có lệnh có giúp gì không.

# H3-C: lệnh đặt TRƯỚC ảnh so với lệnh đặt sau câu hỏi

**Kết quả: Đang chạy.** Đã thử nhỏ trên 5 mô hình, mỗi mô hình 8–9 câu: khi lệnh đặt trước, vi phạm 100% ở mọi mô hình.
Prompt đã được kiểm tra: mô hình vẫn thấy ảnh nhưng lờ lệnh. Đã sinh xong 200 câu cho Qwen; các mô hình khác sinh sau,
rồi chấm khi judge rảnh. V bị bỏ qua vì chưa có ảnh thẻ lệnh.

**Findings mong đợi:**

- Nếu khác biệt lớn: lệnh cấm phai dần theo khoảng cách, hoặc bị câu hỏi phía sau "ghi đè", ở mọi kênh.
- Nếu khác biệt này lại khác nhau giữa các kênh: đó là dấu hiệu về cách attention xử lý token của từng kênh.
- Lưu ý: đặt lệnh "first" cũng là đặt nó trước câu hỏi. Muốn tách hai yếu tố thì cần T_pre, chưa chạy.

# H4 full cho Qwen-7B và Qwen-3B: cùng Delta_A − Delta_T như H4 nhưng nhiều dữ liệu hơn khoảng 4 lần (pilot cho CI vượt 0)

**Kết quả: Đang chạy.** Sinh được 2.884/16.827 câu (7B) và 2.324/16.791 câu (3B), còn khoảng 4,5–5 giờ. Sau đó judge
khoảng 4 giờ mỗi mô hình.

**Findings mong đợi:**

- Nếu CI không chứa 0: có bằng chứng về một gap gõ/nói riêng cho phủ định ở Qwen, và chiều của gap là chiều mà CI
  cho thấy.
- Nếu CI vẫn quanh 0: kết luận trung thực là "Qwen không có gap riêng cho phủ định".

**Vấn đề lớn còn mở:**

- Chưa có người kiểm tra judge trên câu trả lời của B200.
- H2 đã chạy khác với H2 trong tài liệu Idea.
- Phần method (steering, …) đang tắt theo quyết định của Vy.
