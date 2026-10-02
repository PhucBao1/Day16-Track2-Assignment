# Báo cáo LAB 16 — Cloud AI Environment Setup (AWS)

- **Sinh viên:** Nguyễn Phúc Bảo — 2A202602925
- **Cloud / Region:** AWS `us-east-1`
- **Compute node:** `t3.medium` (2 vCPU / 4 GB RAM), Ubuntu 22.04, Python 3.10.12, LightGBM 4.7.0
- **Thời gian `terraform apply` (CPU):** 2 phút 51 giây (27 resources, 13:22:36 → 13:25:27 UTC, 02/10/2026)

## 1. Kết quả LightGBM benchmark (Credit Card Fraud Detection)

Dataset: 284,807 giao dịch, 30 features, tỉ lệ gian lận 0.173%. Chia stratified: train 205,060 / validation 22,785 (early stopping) / test 56,962.
Model: `LGBMClassifier(learning_rate=0.02, num_leaves=31, min_child_samples=50, min_child_weight=1.0)`, early stopping 100 vòng theo AUC.
Số liệu lấy từ `benchmark_result.json`.

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.206 s |
| Thời gian training | 12.612 s |
| Best iteration | 292 |
| AUC-ROC | 0.9777 |
| Accuracy | 0.9996 |
| F1-Score | 0.8729 |
| Precision | 0.9518 |
| Recall | 0.8061 |
| Inference latency (1 row) | 1.281 ms (p95 1.318 ms) |
| Inference throughput (1000 rows) | 10.40 ms (~96,183 rows/s) |

## 2. Tài nguyên & chi phí

- Benchmark output: xem `screenshots/01_benchmark.png`
- CPU / RAM / Network: xem `screenshots/03_top_free_iplink.png`
- EC2 instances đang chạy (t3.medium + t3.micro): xem `screenshots/04a_ec2_instances_running.png`
- Billing: xem `screenshots/04_billing.png` — dịch vụ phát sinh chi phí: EC2, NAT Gateway, ELB
- Ước tính: ~$0.10/giờ (CPU flow)

## 3. Nhận xét (5–10 dòng)

- **Training time:** chỉ ~12.6 s cho 205k dòng × 30 features trên 2 vCPU (`t3.medium`, ~$0.04/giờ) → với dữ liệu dạng bảng cỡ này, CPU nhỏ là đủ, không cần GPU.
- **AUC-ROC 0.978** cho thấy model phân biệt gian lận rất tốt. **Accuracy 99.96% không có nhiều ý nghĩa** vì gian lận chỉ chiếm 0.17% — model đoán "không gian lận" cho mọi giao dịch cũng đạt 99.83%; nên đánh giá bằng AUC, Precision, Recall, F1.
- **Precision 0.95 / Recall 0.81** ở ngưỡng 0.5: ít báo nhầm nhưng bỏ sót ~19% giao dịch gian lận. Với bài toán fraud thường nên hạ ngưỡng để tăng Recall.
- **Tinh chỉnh quan trọng:** với tham số mặc định (`min_child_weight=1e-3`), early stopping dừng ở iteration 1 (AUC 0.927) do lá chứa quá ít mẫu gian lận nhận giá trị cực lớn; tăng `min_child_samples=50`, `min_child_weight=1.0` và giảm `learning_rate=0.02` → 292 iterations, AUC 0.978, F1 0.873.
- **Inference:** ~1.3 ms cho 1 dòng nhưng chỉ ~10.4 ms cho 1000 dòng (~96k dòng/s) → phần lớn latency đơn lẻ là overhead gọi hàm/pandas; batch inference hiệu quả hơn ~120 lần mỗi dòng.
- **Chi phí:** toàn bộ hạ tầng ~$0.10/giờ, trong đó NAT Gateway (~$0.045/giờ) đắt hơn cả compute node — phải `terraform destroy` ngay sau khi làm xong.

## 4. Phụ lục GPU + LLM (tùy chọn)

**Chưa thực hiện** — đang chờ AWS duyệt quota `Running On-Demand G and VT instances` (4 vCPU) tại `us-east-1`.

## 5. Dọn dẹp

- [x] `terraform destroy` → `Destroy complete!` (state trống)
- [x] Kiểm tra `us-east-1` bằng AWS CLI: không còn instance / NAT Gateway / Load Balancer / Elastic IP / VPC của lab
