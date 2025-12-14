import pygame
import numpy as np
import sys
import time

# Simple deterministic GridWorld 4x4
START = (0, 0)
GOAL = (3, 3)
HOLES = {(1, 3), (2, 1), (3, 0), (0, 1)}

def state_of(pos): 
    return pos[0]*4 + pos[1]

actions = [(0,-1),(0,1),(-1,0),(1,0)]  # L,R,U,D
action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN']

def step(pos, a):
    nr = pos[0] + actions[a][0]
    nc = pos[1] + actions[a][1]
    if nr<0 or nr>3 or nc<0 or nc>3:
        return pos, -0.01, False  # Phạt nhẹ khi đâm tường
    npos = (nr,nc)
    if npos in HOLES:
        return npos, -1, True
    if npos == GOAL:
        return npos, 1, True
    return npos, -0.01, False  # Phạt nhẹ mỗi bước để khuyến khích đường ngắn

# Hyperparameters
alpha = 0.1
gamma = 0.95  # Tăng gamma để đánh giá cao reward tương lai
eps = 0.3
max_steps = 100  # Giới hạn số bước mỗi episode

Q = np.zeros((16,4))
episode_rewards = []
episode_steps = []

def eps_greedy(s):
    if np.random.rand() < eps:
        return np.random.randint(4)
    return np.argmax(Q[s])

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1400, 800
GRID_SIZE = 4
CELL_SIZE = 140
GRID_OFFSET_X = 80
GRID_OFFSET_Y = 150

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning GridWorld - Trực Quan Hóa Thuật Toán")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
GREEN = (76, 175, 80)
RED = (244, 67, 54)
BLUE = (33, 150, 243)
YELLOW = (255, 235, 59)
LIGHT_BLUE = (179, 229, 252)
ORANGE = (255, 152, 0)
PURPLE = (156, 39, 176)

# Fonts
font_tiny = pygame.font.Font(None, 18)
font_small = pygame.font.Font(None, 24)
font_medium = pygame.font.Font(None, 30)
font_large = pygame.font.Font(None, 40)
font_title = pygame.font.Font(None, 56)

def draw_grid():
    """Vẽ lưới GridWorld với style đẹp hơn"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = GRID_OFFSET_X + col * CELL_SIZE
            y = GRID_OFFSET_Y + row * CELL_SIZE
            
            pos = (row, col)
            
            # Màu ô
            if pos == GOAL:
                color = GREEN
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 3)
                # Icon goal
                text = font_large.render("★", True, YELLOW)
                screen.blit(text, (x + CELL_SIZE//2 - text.get_width()//2, y + 15))
                label = font_medium.render("GOAL", True, WHITE)
                screen.blit(label, (x + CELL_SIZE//2 - label.get_width()//2, y + 60))
                reward_text = font_small.render("+1.0", True, WHITE)
                screen.blit(reward_text, (x + CELL_SIZE//2 - reward_text.get_width()//2, y + 95))
            elif pos in HOLES:
                color = RED
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 3)
                # Icon hole
                text = font_large.render("✕", True, WHITE)
                screen.blit(text, (x + CELL_SIZE//2 - text.get_width()//2, y + 15))
                label = font_medium.render("HOLE", True, WHITE)
                screen.blit(label, (x + CELL_SIZE//2 - label.get_width()//2, y + 60))
                reward_text = font_small.render("-1.0", True, WHITE)
                screen.blit(reward_text, (x + CELL_SIZE//2 - reward_text.get_width()//2, y + 95))
            elif pos == START:
                color = LIGHT_BLUE
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLUE, (x, y, CELL_SIZE, CELL_SIZE), 3)
                label = font_medium.render("START", True, BLUE)
                screen.blit(label, (x + CELL_SIZE//2 - label.get_width()//2, y + 10))
            else:
                color = WHITE
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 2)

def draw_q_values(highlight_state=None, highlight_action=None):
    """Vẽ Q-values chi tiết cho mỗi ô"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = (row, col)
            if pos in HOLES or pos == GOAL:
                continue
                
            s = state_of(pos)
            x = GRID_OFFSET_X + col * CELL_SIZE
            y = GRID_OFFSET_Y + row * CELL_SIZE
            
            # Highlight ô hiện tại
            if highlight_state is not None and s == highlight_state:
                pygame.draw.rect(screen, ORANGE, (x+3, y+3, CELL_SIZE-6, CELL_SIZE-6), 4)
            
            # Vẽ Q-values cho 4 hướng
            q_vals = Q[s]
            best_action = np.argmax(q_vals)
            
            # Vẽ mũi tên cho action tốt nhất
            arrow_color = BLUE if s == highlight_state else DARK_GRAY
            arrow_size = 25
            center_x = x + CELL_SIZE//2
            center_y = y + CELL_SIZE//2
            
            # Vẽ mũi tên theo action tốt nhất
            if best_action == 0:  # LEFT
                points = [(center_x - arrow_size, center_y), 
                         (center_x - 5, center_y - 10),
                         (center_x - 5, center_y + 10)]
            elif best_action == 1:  # RIGHT
                points = [(center_x + arrow_size, center_y),
                         (center_x + 5, center_y - 10),
                         (center_x + 5, center_y + 10)]
            elif best_action == 2:  # UP
                points = [(center_x, center_y - arrow_size),
                         (center_x - 10, center_y - 5),
                         (center_x + 10, center_y - 5)]
            else:  # DOWN
                points = [(center_x, center_y + arrow_size),
                         (center_x - 10, center_y + 5),
                         (center_x + 10, center_y + 5)]
            
            if highlight_action is not None and s == highlight_state:
                # Highlight action đang thực hiện
                if highlight_action == 0:  # LEFT
                    points = [(center_x - arrow_size, center_y), 
                             (center_x - 5, center_y - 10),
                             (center_x - 5, center_y + 10)]
                    arrow_color = ORANGE
                elif highlight_action == 1:  # RIGHT
                    points = [(center_x + arrow_size, center_y),
                             (center_x + 5, center_y - 10),
                             (center_x + 5, center_y + 10)]
                    arrow_color = ORANGE
                elif highlight_action == 2:  # UP
                    points = [(center_x, center_y - arrow_size),
                             (center_x - 10, center_y - 5),
                             (center_x + 10, center_y - 5)]
                    arrow_color = ORANGE
                else:  # DOWN
                    points = [(center_x, center_y + arrow_size),
                             (center_x - 10, center_y + 5),
                             (center_x + 10, center_y + 5)]
                    arrow_color = ORANGE
            
            pygame.draw.polygon(screen, arrow_color, points)
            
            # Vẽ Q-values ở 4 góc
            positions = [
                (x + 5, y + CELL_SIZE//2 - 10),  # L
                (x + CELL_SIZE - 45, y + CELL_SIZE//2 - 10),  # R
                (x + CELL_SIZE//2 - 20, y + 5),  # U
                (x + CELL_SIZE//2 - 20, y + CELL_SIZE - 20)  # D
            ]
            
            for a in range(4):
                q_val = q_vals[a]
                # Màu theo giá trị Q
                if q_val > 0.5:
                    color = GREEN
                elif q_val < -0.5:
                    color = RED
                else:
                    color = DARK_GRAY
                
                if a == best_action:
                    color = BLUE
                    
                if highlight_action is not None and s == highlight_state and a == highlight_action:
                    color = ORANGE
                
                text = font_tiny.render(f"{q_val:.2f}", True, color)
                screen.blit(text, positions[a])

def draw_agent(pos, color=PURPLE):
    """Vẽ agent với hiệu ứng đẹp hơn"""
    x = GRID_OFFSET_X + pos[1] * CELL_SIZE + CELL_SIZE//2
    y = GRID_OFFSET_Y + pos[0] * CELL_SIZE + CELL_SIZE//2
    pygame.draw.circle(screen, color, (x, y), 25)
    pygame.draw.circle(screen, BLACK, (x, y), 25, 3)
    # Vẽ mắt
    pygame.draw.circle(screen, WHITE, (x - 8, y - 5), 6)
    pygame.draw.circle(screen, WHITE, (x + 8, y - 5), 6)
    pygame.draw.circle(screen, BLACK, (x - 8, y - 5), 3)
    pygame.draw.circle(screen, BLACK, (x + 8, y - 5), 3)

def draw_info(episode, total_reward, step_count, current_action, learning_info):
    """Vẽ thông tin huấn luyện chi tiết"""
    info_x = GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 80
    
    # Title
    title = font_title.render("Q-LEARNING", True, BLUE)
    screen.blit(title, (info_x - 20, 40))
    
    # Divider
    pygame.draw.line(screen, GRAY, (info_x - 20, 95), (info_x + 350, 95), 2)
    
    # Hyperparameters box
    y = 120
    pygame.draw.rect(screen, LIGHT_BLUE, (info_x - 10, y - 10, 360, 180), border_radius=10)
    pygame.draw.rect(screen, BLUE, (info_x - 10, y - 10, 360, 180), 2, border_radius=10)
    
    header = font_medium.render("THAM SỐ", True, BLUE)
    screen.blit(header, (info_x, y))
    y += 40
    
    params = [
        f"Alpha (α) - Tốc độ học: {alpha}",
        f"Gamma (γ) - Chiết khấu: {gamma}",
        f"Epsilon (ε) - Khám phá: {eps}",
        f"Max Steps/Episode: {max_steps}",
    ]
    
    for param in params:
        text = font_small.render(param, True, BLACK)
        screen.blit(text, (info_x + 10, y))
        y += 30
    
    # Training info box
    y += 20
    pygame.draw.rect(screen, (255, 248, 220), (info_x - 10, y - 10, 360, 230), border_radius=10)
    pygame.draw.rect(screen, ORANGE, (info_x - 10, y - 10, 360, 230), 2, border_radius=10)
    
    header = font_medium.render("TRẠNG THÁI HUẤN LUYỆN", True, ORANGE)
    screen.blit(header, (info_x, y))
    y += 40
    
    training_info = [
        f"Episode: {episode}",
        f"Số bước: {step_count}",
        f"Tổng reward: {total_reward:.3f}",
        f"Hành động: {current_action if current_action else 'Chưa bắt đầu'}",
    ]
    
    for info in training_info:
        text = font_small.render(info, True, BLACK)
        screen.blit(text, (info_x + 10, y))
        y += 35
    
    # Learning details
    if learning_info:
        y += 10
        details = [
            f"Q cũ: {learning_info['old_q']:.3f}",
            f"Q mới: {learning_info['new_q']:.3f}",
            f"TD Error: {learning_info['td_error']:.3f}",
        ]
        for detail in details:
            text = font_tiny.render(detail, True, DARK_GRAY)
            screen.blit(text, (info_x + 10, y))
            y += 22
    
    # Statistics box
    y += 20
    if len(episode_rewards) > 0:
        pygame.draw.rect(screen, (240, 255, 240), (info_x - 10, y - 10, 360, 120), border_radius=10)
        pygame.draw.rect(screen, GREEN, (info_x - 10, y - 10, 360, 120), 2, border_radius=10)
        
        header = font_medium.render("THỐNG KÊ", True, GREEN)
        screen.blit(header, (info_x, y))
        y += 40
        
        avg_reward = np.mean(episode_rewards[-100:])
        avg_steps = np.mean(episode_steps[-100:])
        
        stats = [
            f"TB Reward (100 eps): {avg_reward:.3f}",
            f"TB Steps (100 eps): {avg_steps:.1f}",
        ]
        
        for stat in stats:
            text = font_small.render(stat, True, BLACK)
            screen.blit(text, (info_x + 10, y))
            y += 30
    
    # Controls box
    y += 30
    pygame.draw.rect(screen, (245, 245, 245), (info_x - 10, y - 10, 360, 180), border_radius=10)
    pygame.draw.rect(screen, DARK_GRAY, (info_x - 10, y - 10, 360, 180), 2, border_radius=10)
    
    header = font_medium.render("ĐIỀU KHIỂN", True, DARK_GRAY)
    screen.blit(header, (info_x, y))
    y += 40
    
    controls = [
        "SPACE - Bước tiếp theo",
        "ENTER - Tự động chạy",
        "R - Reset toàn bộ",
        "ESC - Thoát",
    ]
    
    for control in controls:
        text = font_small.render(control, True, BLACK)
        screen.blit(text, (info_x + 10, y))
        y += 30

def draw_path(path):
    """Vẽ đường đi của agent"""
    if len(path) < 2:
        return
    
    for i in range(len(path) - 1):
        pos1 = path[i]
        pos2 = path[i + 1]
        
        x1 = GRID_OFFSET_X + pos1[1] * CELL_SIZE + CELL_SIZE//2
        y1 = GRID_OFFSET_Y + pos1[0] * CELL_SIZE + CELL_SIZE//2
        x2 = GRID_OFFSET_X + pos2[1] * CELL_SIZE + CELL_SIZE//2
        y2 = GRID_OFFSET_Y + pos2[0] * CELL_SIZE + CELL_SIZE//2
        
        pygame.draw.line(screen, PURPLE, (x1, y1), (x2, y2), 4)
        pygame.draw.circle(screen, PURPLE, (x2, y2), 6)

# Training loop với visualization
running = True
auto_run = False
episode = 0
pos = START
s = state_of(pos)
done = False
step_count = 0
total_reward = 0
current_action = None
learning_info = None
path = [pos]
max_episodes = 5000

while running and episode < max_episodes:
    clock.tick(60 if not auto_run else 10)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE and not auto_run:
                # Single step
                if not done:
                    a = eps_greedy(s)
                    current_action = action_names[a]
                    old_q = Q[s, a]
                    
                    npos, r, done = step(pos, a)
                    ns = state_of(npos)
                    
                    # Q-learning update
                    target = r + gamma * np.max(Q[ns])
                    td_error = target - old_q
                    Q[s,a] = old_q + alpha * td_error
                    
                    learning_info = {
                        'old_q': old_q,
                        'new_q': Q[s, a],
                        'td_error': td_error
                    }
                    
                    total_reward += r
                    step_count += 1
                    pos = npos
                    s = ns
                    path.append(pos)
                    
                    if done or step_count >= max_steps:
                        episode_rewards.append(total_reward)
                        episode_steps.append(step_count)
                else:
                    # Start new episode
                    episode += 1
                    pos = START
                    s = state_of(pos)
                    done = False
                    step_count = 0
                    total_reward = 0
                    current_action = None
                    learning_info = None
                    path = [pos]
            elif event.key == pygame.K_RETURN:
                auto_run = not auto_run
            elif event.key == pygame.K_r:
                Q = np.zeros((16,4))
                episode = 0
                pos = START
                s = state_of(pos)
                done = False
                step_count = 0
                total_reward = 0
                current_action = None
                learning_info = None
                episode_rewards = []
                episode_steps = []
                path = [pos]
    
    if auto_run:
        if not done and step_count < max_steps:
            a = eps_greedy(s)
            current_action = action_names[a]
            old_q = Q[s, a]
            
            npos, r, done = step(pos, a)
            ns = state_of(npos)
            
            target = r + gamma * np.max(Q[ns])
            td_error = target - old_q
            Q[s,a] = old_q + alpha * td_error
            
            learning_info = {
                'old_q': old_q,
                'new_q': Q[s, a],
                'td_error': td_error
            }
            
            total_reward += r
            step_count += 1
            pos = npos
            s = ns
            path.append(pos)
            
            if done or step_count >= max_steps:
                episode_rewards.append(total_reward)
                episode_steps.append(step_count)
        else:
            episode += 1
            pos = START
            s = state_of(pos)
            done = False
            step_count = 0
            total_reward = 0
            current_action = None
            learning_info = None
            path = [pos]
    
    # Drawing
    screen.fill(WHITE)
    
    # Draw title
    title = font_title.render("GRIDWORLD Q-LEARNING VISUALIZATION", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    draw_grid()
    
    if episode > 0:
        draw_q_values(s if not done else None, 
                     eps_greedy(s) if not done and current_action else None)
    
    draw_path(path)
    
    if not done:
        draw_agent(pos)
    
    draw_info(episode, total_reward, step_count, current_action, learning_info)
    
    pygame.display.flip()

pygame.quit()
sys.exit()