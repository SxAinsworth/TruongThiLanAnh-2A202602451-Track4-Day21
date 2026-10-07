# Báo cáo Day 6: Kiểm tra calibration LiDAR-camera bằng projection

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** [Trương Thị Lan Anh]
- **MSSV:** [2A202602451] (phải trùng với MSSV trong tên repo `<HoVaTen>-<MSSV>-Track4-Day21`)
- **Lớp:** []
- **Link repo:** [https://github.com/SxAinsworth/TruongThiLanAnh-2A202602451-Track4-Day21]
- **Topic:** [E] — [Dashboard sức khoẻ dữ liệu]
- **Dataset:** data/kitti_mini (chính), data/nuscenes_mini_subset (so sánh), data/synthetic (debug)
- **Các frame đã dùng:** [ĐIỀN danh sách frame id, ví dụ 000011, 000049 hoặc scene-0103_010]

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

*(Nháp CP1)* Trên `kitti_mini`, lệch yaw 1° làm hơn 10% điểm LiDAR của xe ở xa (≥ 30 m) rơi ra khỏi 2D box của xe, trong khi xe ở gần (< 15 m) mất dưới 5%. Lệch tịnh tiến 5 cm cho xu hướng ngược lại: ảnh hưởng xe gần nhiều hơn xe xa.

## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
