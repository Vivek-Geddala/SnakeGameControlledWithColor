import pygame
import random
import sys
import cv2
import numpy as np

# ------------- Pygame Setup -------------
pygame.init()

WIDTH = 600
HEIGHT = 400
BLOCK_SIZE = 20
FPS = 5   # base speed

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Color-Controlled Snake - Chinnu 🐍")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 24)

# ------------- Camera Setup (OpenCV) -------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)


def draw_text(text, color, x, y):
    label = font.render(text, True, color)
    screen.blit(label, (x, y))


def get_direction_from_color(current_dx, current_dy):
    """
    Use a BLUE-colored object to control direction.

    - Show blue object in front of camera.
    - If its position is left/right/up/down relative to center,
      we change the snake's direction.

    If no blue object is detected, keep current direction.
    """
    ret, frame = cap.read()
    if not ret:
        return current_dx, current_dy

    # Mirror effect (like a selfie)
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- BLUE color range in HSV (you can tune this if needed) ---
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours of the blue area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = frame.shape
    frame_center_x = width // 2
    frame_center_y = height // 2

    direction_dx = current_dx
    direction_dy = current_dy

    if contours:
        # Take the largest blue object
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 1000:  # ignore tiny noise blobs
            (x, y, w, h) = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2

            # Draw center + rectangle for debugging
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.circle(frame, (frame_center_x, frame_center_y), 7, (0, 255, 0), -1)

            # Difference from center
            dx = cx - frame_center_x
            dy = cy - frame_center_y

            threshold = 40  # how much movement needed to change direction

            # Decide if movement is more horizontal or vertical
            if abs(dx) > abs(dy):
                # Left / Right
                if dx > threshold and current_dx == 0:
                    direction_dx = BLOCK_SIZE
                    direction_dy = 0
                elif dx < -threshold and current_dx == 0:
                    direction_dx = -BLOCK_SIZE
                    direction_dy = 0
            else:
                # Up / Down
                if dy > threshold and current_dy == 0:
                    direction_dy = BLOCK_SIZE
                    direction_dx = 0
                elif dy < -threshold and current_dy == 0:
                    direction_dy = -BLOCK_SIZE
                    direction_dx = 0

    # Show camera with view (for debugging)
    cv2.imshow("Show a BLUE object to control", frame)

    # If user presses ESC in the camera window, close that window (game continues)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyWindow("Show a BLUE object to control")

    return direction_dx, direction_dy


def game_loop():
    x = WIDTH // 2
    y = HEIGHT // 2

    dx = 0
    dy = 0

    snake_body = []
    snake_length = 1

    food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
    food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE

    # --- Heartbeat animation variables for food ---
    base_radius = BLOCK_SIZE // 2              # normal size of the dot
    food_radius = float(base_radius)           # current animated radius
    radius_min = base_radius * 0.6             # smallest size
    radius_max = base_radius * 1.4             # biggest size
    radius_step = 0.3                          # how fast it grows/shrinks
    growing = True                             # direction of pulse

    score = 0
    game_over = False

    global FPS

    while True:
        while game_over:
            screen.fill(BLACK)
            draw_text("Game Over! Press ENTER to Play Again or ESC to Quit", RED, 20, HEIGHT // 2 - 20)
            draw_text(f"Score: {score}", YELLOW, WIDTH // 2 - 40, HEIGHT // 2 + 20)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return  # restart game
                    if event.key == pygame.K_ESCAPE:
                        cap.release()
                        cv2.destroyAllWindows()
                        pygame.quit()
                        sys.exit()

        # --- Keyboard backup control (optional) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and dx == 0:
                    dx = -BLOCK_SIZE
                    dy = 0
                elif event.key == pygame.K_RIGHT and dx == 0:
                    dx = BLOCK_SIZE
                    dy = 0
                elif event.key == pygame.K_UP and dy == 0:
                    dy = -BLOCK_SIZE
                    dx = 0
                elif event.key == pygame.K_DOWN and dy == 0:
                    dy = BLOCK_SIZE
                    dx = 0

        # --- Color control with BLUE object ---
        dx, dy = get_direction_from_color(dx, dy)

        # Move snake
        if dx != 0 or dy != 0:
            x += dx
            y += dy

        
        # Wrap-around instead of wall collision
        if x < 0:
            x = WIDTH - BLOCK_SIZE
        elif x >= WIDTH:
            x = 0

        if y < 0:
            y = HEIGHT - BLOCK_SIZE
        elif y >= HEIGHT:
            y = 0

        

        # Update snake body
        head = [x, y]
        snake_body.append(head)
        if len(snake_body) > snake_length:
            del snake_body[0]

        # Self collision
        for block in snake_body[:-1]:
            if block == head:
                game_over = True

        # Eat food (grid based, animation doesn't affect this)
        if x == food_x and y == food_y:
            snake_length += 1
            score += 1
            FPS += 0.5  # speed increase

            food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
            food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE

            # reset heartbeat size when new food appears
            food_radius = float(base_radius)
            growing = True

        # -------- Drawing --------
        screen.fill(BLACK)

        # Round snake body
        for block in snake_body:
            center_x = block[0] + BLOCK_SIZE // 2
            center_y = block[1] + BLOCK_SIZE // 2
            radius = BLOCK_SIZE // 2
            pygame.draw.circle(screen, GREEN, (center_x, center_y), radius)

        # --- Heartbeat effect for food: always visible, pulsing size ---
        if growing:
            food_radius += radius_step
            if food_radius >= radius_max:
                food_radius = radius_max
                growing = False
        else:
            food_radius -= radius_step
            if food_radius <= radius_min:
                food_radius = radius_min
                growing = True

        food_center_x = food_x + BLOCK_SIZE // 2
        food_center_y = food_y + BLOCK_SIZE // 2

        pygame.draw.circle(
            screen,
            YELLOW,
            (food_center_x, food_center_y),
            int(food_radius)
        )

        # Score & instruction
        draw_text(f"Score: {score}", WHITE, 10, 10)
        draw_text("Use BLUE object: LEFT / RIGHT / UP / DOWN", WHITE, 10, HEIGHT - 30)

        pygame.display.flip()
        clock.tick(int(FPS))


def main():
    try:
        while True:
            game_loop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
