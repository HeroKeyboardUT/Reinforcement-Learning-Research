# Reinforcement Learning Research Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-Educational-red.svg)

**Đại học Bách Khoa TP.HCM - Khoa Khoa học và Kỹ thuật Máy tính**

</div>

---

## Tổng quan

Đồ án này trình bày việc nghiên cứu và triển khai các thuật toán **Học Tăng Cường (Reinforcement Learning)** thông qua hai bài toán thực tế:

| #   | Project              | Thuật toán | Mô tả                                          |
| --- | -------------------- | ---------- | ---------------------------------------------- |
| 1   | **Grid World**       | Q-Learning | Bài toán tìm đường cổ điển "Chuột tìm pho mát" |
| 2   | **BipedalWalker-v3** | PPO        | Huấn luyện robot hai chân vượt địa hình        |

---

## Cơ sở lý thuyết

### Reinforcement Learning là gì?

Học Tăng Cường là một nhánh của Machine Learning trong đó:

```
Agent ←→ Environment
  ↓           ↓
Action → State + Reward
```

- **Agent** (Tác nhân): Đưa ra quyết định
- **Environment** (Môi trường): Phản hồi lại hành động
- **Policy** (Chính sách): Chiến lược chọn hành động
- **Reward** (Phần thưởng): Tín hiệu đánh giá hành động

### Lịch sử phát triển (Mục 2 - Báo cáo)

Như trình bày trong **Dòng chảy "Học qua Thử và Sai"**:

| Thập kỷ | Cột mốc quan trọng                  |
| ------- | ----------------------------------- |
| 1950s   | Bellman đề xuất Dynamic Programming |
| 1989    | Watkins phát minh Q-Learning        |
| 2013    | DeepMind giới thiệu DQN             |
| 2017    | PPO trở thành chuẩn mực             |
| 2023    | DreamerV3 và World Models           |

---

## Project 1: Grid World Environment

### Mô tả bài toán

Môi trường lưới **4x4** với 16 trạng thái, nơi tác nhân phải tìm đường từ **START** đến **GOAL** mà không rơi vào **HOLE**.

```
┌─────┬─────┬─────┬─────┐
│START│HOLE │     │     │
├─────┼─────┼─────┼─────┤
│     │     │     │HOLE │
├─────┼─────┼─────┼─────┤
│     │HOLE │     │     │
├─────┼─────┼─────┼─────┤
│HOLE │     │     │GOAL │
└─────┴─────┴─────┴─────┘
```

### Thành phần MDP (Markov Decision Process)

| Thành phần      | Chi tiết                                   |
| --------------- | ------------------------------------------ |
| **States (S)**  | 16 ô vuông (0-15)                          |
| **Actions (A)** | 4 hướng: LEFT, RIGHT, UP, DOWN             |
| **Rewards (R)** | +1.0 (Goal), -1.0 (Hole), -0.01 (mỗi bước) |
| **Transition**  | Deterministic (xác định)                   |

### Cài đặt và Chạy

```bash
# Clone repository
git clone <repository-url>
cd DATH

# Cài đặt dependencies
pip install pygame numpy

# Chạy chương trình
python grid.py
```

### Điều khiển

| Phím    | Chức năng              |
| ------- | ---------------------- |
| `SPACE` | Thực hiện một bước học |
| `ENTER` | Bật/tắt chế độ tự động |
| `R`     | Reset toàn bộ Q-Table  |
| `ESC`   | Thoát chương trình     |

### Tham số Hyperparameters

```python
alpha = 0.1      # Tốc độ học (Learning Rate)
gamma = 0.95     # Hệ số chiết khấu (Discount Factor)
eps = 0.3        # Xác suất khám phá (Epsilon-greedy)
max_steps = 100  # Giới hạn bước mỗi episode
```

### Chiến lược Epsilon-Greedy (Mục 3.1.2)

Theo báo cáo, tác nhân sử dụng chiến lược cân bằng giữa:

- **Khám phá (Exploration):** Chọn ngẫu nhiên hành động với xác suất ε = 0.3
- **Khai thác (Exploitation):** Chọn hành động có Q-value cao nhất với xác suất 1-ε = 0.7

### Công thức cập nhật Q-Learning

```
Q(s,a) ← Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]
                        └──────────────────────────┘
                              TD Error (δ)
```

### Kết quả thực nghiệm (Bảng 1 - Báo cáo)

Sau **10.000 episodes** huấn luyện, Q-Table học được các giá trị:

| State | Vị trí | Hành động tối ưu | Q-Value | "Tư duy" của AI              |
| ----- | ------ | ---------------- | ------- | ---------------------------- |
| 0     | Start  | DOWN             | 0.590   | "Đường này điểm cao nhất"    |
| 4     | (1,0)  | DOWN             | 0.686   | "Tiếp tục xuống là an toàn"  |
| 5     | (1,1)  | RIGHT            | 0.729   | "Rẽ phải là đường sáng nhất" |
| 6     | (1,2)  | DOWN             | 0.806   | "Tránh hố, đi xuống"         |
| 10    | (2,2)  | RIGHT            | 0.855   | "Đang đến gần đích rồi"      |
| 14    | (3,2)  | RIGHT            | 1.000   | "Nhìn thấy kho báu rồi!"     |

---

## Project 2: BipedalWalker-v3 (PPO)

### Mô tả bài toán (Mục 3.2)

Huấn luyện robot hai chân học cách đi bộ và vượt địa hình bằng thuật toán **Proximal Policy Optimization (PPO)**.

Notebook Link: https://www.kaggle.com/code/phmquanghiu/doantonghop1

Notebook Link: https://www.kaggle.com/code/phmquanghiu/doantonghop

### Không gian trạng thái (24 chiều)

| Index | Thông tin             | Mô tả                   |
| ----- | --------------------- | ----------------------- |
| 0     | Hull Angle            | Góc nghiêng thân        |
| 1     | Hull Angular Velocity | Vận tốc góc             |
| 2-3   | Velocity (x, y)       | Vận tốc ngang/dọc       |
| 4-13  | Joint States          | Góc và vận tốc các khớp |
| 14-23 | LIDAR                 | 10 cảm biến khoảng cách |

### Không gian hành động (4 chiều liên tục)

| Index | Khớp      | Phạm vi | Mô tả                   |
| ----- | --------- | ------- | ----------------------- |
| 0     | Hông trái | [-1, 1] | Điều khiển đùi trái     |
| 1     | Gối trái  | [-1, 1] | Duỗi/gập cẳng chân trái |
| 2     | Hông phải | [-1, 1] | Điều khiển đùi phải     |
| 3     | Gối phải  | [-1, 1] | Duỗi/gập cẳng chân phải |

### Hệ thống Reward

```
Reward = Tiến về phía trước (+)
       - Chi phí năng lượng (-)
       - Phạt ngã (-100)
       + Bonus hoàn thành (+300)
```

### Quá trình hội tụ (Hình 10 - Báo cáo)

| Giai đoạn      | Episodes | Reward     | Mô tả              |
| -------------- | -------- | ---------- | ------------------ |
| **Bùng nổ**    | 0-200    | -100 → +50 | Robot "xóa mù chữ" |
| **Tinh chỉnh** | 200-1000 | Dao động   | Giữa ~300 và ~-100 |
| **Ổn định**    | >1000    | ~250-300   | Đi vững vàng       |

---

## Các thuật toán nâng cao (Mục 4)

### DreamerV3 và World Models

Như trình bày trong báo cáo, các thuật toán hiện đại sử dụng:

- **World Model**: Mô hình nội tại về môi trường
- **Imagination**: Học trong "giấc mơ" mà không cần tương tác thực
- **Sample Efficiency**: Hiệu quả mẫu cao hơn

---

## Ứng dụng thực tiễn (Mục 5)

| Lĩnh vực       | Ứng dụng         | Ví dụ                  |
| -------------- | ---------------- | ---------------------- |
| **Game AI**    | NPC thông minh   | AlphaGo, OpenAI Five   |
| **Robotics**   | Điều khiển robot | Boston Dynamics        |
| **Finance**    | Trading tự động  | Portfolio Optimization |
| **Healthcare** | Tối ưu điều trị  | Drug Discovery         |

---

## Thách thức và Xu hướng (Mục 6)

### Thách thức hiện tại

1. **Sample Inefficiency** - Cần nhiều dữ liệu huấn luyện
2. **Reward Shaping** - Thiết kế hàm reward khó
3. **Sim-to-Real Gap** - Khác biệt mô phỏng vs thực tế
4. **Safety** - Đảm bảo an toàn khi triển khai

### Xu hướng tương lai

- **Offline RL** - Học từ dữ liệu có sẵn
- **Multi-Agent RL** - Nhiều agent phối hợp
- **Foundation Models** - Mô hình nền tảng cho RL

---

## Cấu trúc thư mục

```
DATH/
├── 📄 grid.py              # Project 1: Q-Learning GridWorld
├── 📄 README.md            # File hướng dẫn này
├── 📁 Report/
    ├── 📄 main.tex         # Báo cáo LaTeX chính
    ├── 📄 main.pdf         # Báo cáo PDF đã biên dịch
    ├── 📄 references.bib   # Tài liệu tham khảo

```

---

## Tài liệu tham khảo

1. Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.

2. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8(3-4), 279-292.

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). _Proximal Policy Optimization Algorithms_. arXiv:1707.06347.

4. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). _Mastering Diverse Domains through World Models_. arXiv:2301.04104.

5. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518(7540), 529-533.

---

## Ghi chú

- Đồ án được thực hiện cho mục đích **học tập** tại Đại học Bách Khoa TP.HCM
- Code được viết bằng **Python 3.8+**
- Visualization sử dụng **Pygame** để trực quan hóa quá trình học

---

<div align="center">

**© 2024 - Phạm Quang Hiếu - ĐHBK TP.HCM**

_"The only way to learn is by doing."_ - Richard Sutton

</div>
