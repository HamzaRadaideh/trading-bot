import pygame
import math
from queue import PriorityQueue

# Initialize Pygame
pygame.init()

# Constants (floats where possible)
WINDOW_SIZE = (800.0, 600.0)
PLAYER_SIZE = 20.0
OBSTACLE_COLOR = (100, 100, 100)
PLAYER_COLOR = (0, 255, 0)
PATH_COLOR = (255, 0, 0)
GRID_SIZE = 40.0  # float-based grid
MOVE_SPEED = 2.0
BACKGROUND_COLOR = (255, 255, 255)
CLICK_COOLDOWN = 0.1  # seconds
SEARCH_RADIUS = 5.0   # how many grid cells to search around a blocked target

# Camera constants
world_size = (1600.0, 1200.0)  # The actual size of your game world (floats)
CAMERA_SPEED = 5.0            # float speed
EDGE_MARGIN = 50.0            # float threshold near screen edges

class Game:
    def __init__(self):
        # Use int() only where absolutely required: set_mode demands integer sizes
        self.screen = pygame.display.set_mode((int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1])))
        pygame.display.set_caption("Top-down Float Movement Game + Camera Edge Scroll (pygame-ce floats)")
        self.clock = pygame.time.Clock()
        
        # Player properties (floats)
        self.player_pos = [world_size[0] / 2.0, world_size[1] / 2.0]
        self.target_pos = None
        self.current_path = []  # list of float (x, y) waypoints
        self.is_moving = False
        
        # Obstacles as float-based FRects (available in pygame-ce)
        self.obstacles = [
            pygame.FRect(300.0, 200.0, 50.0, 200.0),
            pygame.FRect(500.0, 100.0, 50.0, 200.0),
            pygame.FRect(200.0, 400.0, 200.0, 50.0)
        ]

        # Camera offset as floats
        self.camera_offset = [0.0, 0.0]

        # Other helpers
        self.message = ""
        self.last_click_time = 0.0

    def line_intersects_obstacle(self, start, end):
        """
        Check if a line segment (start -> end) intersects any float-based obstacle.
        start and end are (x, y) floats.
        """
        for obstacle in self.obstacles:
            # Quick check if either endpoint is inside the obstacle
            if obstacle.collidepoint(start) or obstacle.collidepoint(end):
                return True

            x1, y1 = start
            x2, y2 = end

            # Represent the obstacle edges in float
            rect_lines = [
                ((obstacle.left,  obstacle.top),    (obstacle.right, obstacle.top)),
                ((obstacle.right, obstacle.top),    (obstacle.right, obstacle.bottom)),
                ((obstacle.right, obstacle.bottom), (obstacle.left,  obstacle.bottom)),
                ((obstacle.left,  obstacle.bottom), (obstacle.left,  obstacle.top))
            ]

            # Check intersection with each edge
            for line_start, line_end in rect_lines:
                x3, y3 = line_start
                x4, y4 = line_end

                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                # If nearly zero, lines are parallel or coincident
                if abs(denominator) < 1e-9:
                    continue

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

                # If 0 <= t <= 1 and 0 <= u <= 1, line segments intersect
                if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                    return True
        return False

    def smooth_path(self, path, max_turn=0.5):
        """
        Attempt to shorten the path by skipping intermediate points
        if there's direct line-of-sight without hitting obstacles.
        path is a list of float-based (x, y) positions.
        """
        if len(path) < 3:
            return path

        smoothed = [path[0]]  # always keep the start
        for i in range(1, len(path) - 1):
            prev_pt = smoothed[-1]
            curr_pt = path[i]
            next_pt = path[i + 1]

            # Check if direct path to next point is blocked
            if self.line_intersects_obstacle(prev_pt, next_pt):
                smoothed.append(curr_pt)
                continue

            # Check angle between segments
            v1 = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
            v2 = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])

            len_v1 = math.hypot(*v1)
            len_v2 = math.hypot(*v2)
            if len_v1 == 0.0 or len_v2 == 0.0:
                smoothed.append(curr_pt)
                continue

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = dot / (len_v1 * len_v2)

            # If angle is too sharp, keep curr_pt
            if cos_angle < max_turn:
                smoothed.append(curr_pt)

        smoothed.append(path[-1])  # always keep the end
        return smoothed

    def move_player(self):
        """Move the player along the current path if one exists, using float-based movement."""
        if not self.current_path:
            self.is_moving = False
            return

        if len(self.current_path) < 2:
            self.current_path = []
            self.is_moving = False
            return

        # The next waypoint
        target = self.current_path[1]
        dx = target[0] - self.player_pos[0]
        dy = target[1] - self.player_pos[1]
        distance = math.hypot(dx, dy)

        # Small "dead zone" to avoid jitter
        if distance < (MOVE_SPEED + 1.0):
            # We reached the current waypoint
            self.current_path.pop(0)
            if len(self.current_path) < 2:
                self.is_moving = False
            return
        else:
            # Move towards the next waypoint
            angle = math.atan2(dy, dx)
            new_x = self.player_pos[0] + MOVE_SPEED * math.cos(angle)
            new_y = self.player_pos[1] + MOVE_SPEED * math.sin(angle)

            # Check if movement path intersects any obstacle
            if not self.line_intersects_obstacle(
                (self.player_pos[0], self.player_pos[1]),
                (new_x, new_y)
            ):
                self.player_pos[0] = new_x
                self.player_pos[1] = new_y
            else:
                # If blocked, stop
                self.is_moving = False

    def handle_click(self, world_x, world_y):
        """
        Handle a click in world coordinates (floats),
        setting a new path if needed.
        """
        new_target = (float(world_x), float(world_y))

        # Check distance to final waypoint of current path
        if self.current_path:
            last_waypoint = self.current_path[-1]
            dist_to_last_waypoint = math.hypot(
                new_target[0] - last_waypoint[0],
                new_target[1] - last_waypoint[1]
            )
            # If the new click is too close to the final waypoint, ignore
            if dist_to_last_waypoint < GRID_SIZE:
                return

        # Also ignore if it's too close to the current target
        if self.target_pos:
            dist_to_current = math.hypot(
                new_target[0] - self.target_pos[0],
                new_target[1] - self.target_pos[1]
            )
            if dist_to_current < GRID_SIZE:
                return

        self.target_pos = new_target
        
        # Recalculate path
        self.current_path = self.find_path(self.player_pos, self.target_pos)
        self.is_moving = len(self.current_path) > 0

    def get_grid_pos(self, pos):
        """
        Convert a float (x, y) into discrete grid cell (gx, gy).
        This is the minimal place we must use int, because we want a grid index.
        """
        gx = int(pos[0] // GRID_SIZE)
        gy = int(pos[1] // GRID_SIZE)
        return (gx, gy)

    def get_pixel_pos(self, grid_pos):
        """
        Convert a grid cell (gx, gy) back to the center of that cell in float.
        """
        return (
            grid_pos[0] * GRID_SIZE + (GRID_SIZE / 2.0),
            grid_pos[1] * GRID_SIZE + (GRID_SIZE / 2.0)
        )

    def is_valid_position(self, grid_pos):
        """
        Check whether a grid cell is valid (inside the world and not colliding).
        grid_pos is (gx, gy) in integers, again absolutely necessary for discrete pathfinding.
        """
        gx, gy = grid_pos
        max_gx = int(world_size[0] // GRID_SIZE)
        max_gy = int(world_size[1] // GRID_SIZE)

        # If outside the world, invalid
        if gx < 0 or gy < 0 or gx >= max_gx or gy >= max_gy:
            return False

        # Make a float-based test rect for collision
        pixel_x = gx * GRID_SIZE
        pixel_y = gy * GRID_SIZE
        test_rect = pygame.FRect(pixel_x, pixel_y, GRID_SIZE, GRID_SIZE)

        for obstacle in self.obstacles:
            if test_rect.colliderect(obstacle):
                return False
        return True

    def find_nearest_valid_target(self, target_grid_pos):
        """
        Find the nearest valid grid position if the given one is blocked.
        target_grid_pos is (gx, gy).
        """
        if self.is_valid_position(target_grid_pos):
            return target_grid_pos

        tx, ty = target_grid_pos
        best_pos = None
        min_dist = float('inf')

        # Spiral outwards up to SEARCH_RADIUS
        # (We keep these as floats, but grid offsets must be int)
        r_max = int(SEARCH_RADIUS)
        for r in range(1, r_max + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Only check the perimeter of the ring for efficiency
                    if abs(dx) == r or abs(dy) == r:
                        test_pos = (tx + dx, ty + dy)
                        if self.is_valid_position(test_pos):
                            dist = math.hypot(dx, dy)
                            if dist < min_dist:
                                min_dist = dist
                                best_pos = test_pos

            if best_pos is not None:
                return best_pos

        return None

    def get_neighbors(self, grid_pos):
        """
        Get valid 8-directional neighbors for A* pathfinding.
        grid_pos is (gx, gy) in ints.
        """
        gx, gy = grid_pos
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (-1, 1), (1, -1), (-1, -1)
        ]
        for dx, dy in directions:
            new_pos = (gx + dx, gy + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def heuristic(self, a, b):
        """
        Heuristic for A* (Euclidean distance in grid coordinates).
        a, b are (gx, gy) in integers.
        """
        return math.hypot((b[0] - a[0]), (b[1] - a[1]))

    def find_path(self, start_pixel, goal_pixel):
        """
        Use A* to find a path from a float start_pixel to a float goal_pixel.
        Returns a list of float (x, y) path points.
        """
        start_g = self.get_grid_pos(start_pixel)
        goal_g = self.get_grid_pos(goal_pixel)

        # If goal cell is invalid, find the nearest valid cell
        if not self.is_valid_position(goal_g):
            nearest_goal = self.find_nearest_valid_target(goal_g)
            if nearest_goal is None:
                self.message = "No valid path possible!"
                return []
            goal_g = nearest_goal
            self.message = "Moving to nearest reachable position"

        frontier = PriorityQueue()
        frontier.put((0.0, start_g))
        came_from = {start_g: None}
        cost_so_far = {start_g: 0.0}

        while not frontier.empty():
            current = frontier.get()[1]
            if current == goal_g:
                break

            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1.0
                if (next_pos not in cost_so_far) or (new_cost < cost_so_far[next_pos]):
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal_g, next_pos)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current

        # If we never reached the goal
        if goal_g not in came_from:
            self.message = "No path found!"
            return []

        # Reconstruct path in float space
        path = []
        current = goal_g
        while current is not None:
            path.append(self.get_pixel_pos(current))
            current = came_from[current]
        path.reverse()

        # Optionally smooth
        path = self.smooth_path(path)
        return path

    def update_camera(self, dt):
        """Move camera using floats when the mouse is near screen edges."""
        # mouse.get_pos() returns int, we convert to float for consistency
        mx, my = pygame.mouse.get_pos()
        mouse_x, mouse_y = float(mx), float(my)

        # Move camera left
        if mouse_x < EDGE_MARGIN:
            self.camera_offset[0] -= CAMERA_SPEED
        # Move camera right
        elif mouse_x > (WINDOW_SIZE[0] - EDGE_MARGIN):
            self.camera_offset[0] += CAMERA_SPEED

        # Move camera up
        if mouse_y < EDGE_MARGIN:
            self.camera_offset[1] -= CAMERA_SPEED
        # Move camera down
        elif mouse_y > (WINDOW_SIZE[1] - EDGE_MARGIN):
            self.camera_offset[1] += CAMERA_SPEED

        # Clamp camera to world bounds
        max_cam_x = world_size[0] - WINDOW_SIZE[0]
        max_cam_y = world_size[1] - WINDOW_SIZE[1]

        if self.camera_offset[0] < 0.0:
            self.camera_offset[0] = 0.0
        if self.camera_offset[1] < 0.0:
            self.camera_offset[1] = 0.0

        if self.camera_offset[0] > max_cam_x:
            self.camera_offset[0] = max_cam_x
        if self.camera_offset[1] > max_cam_y:
            self.camera_offset[1] = max_cam_y

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            current_time = pygame.time.get_ticks() / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Left click = 1
                    if event.button == 1:
                        if (current_time - self.last_click_time) >= CLICK_COOLDOWN:
                            # Convert these to float; they're initially int
                            mouse_world_x = float(event.pos[0]) + self.camera_offset[0]
                            mouse_world_y = float(event.pos[1]) + self.camera_offset[1]
                            self.handle_click(mouse_world_x, mouse_world_y)
                            self.last_click_time = current_time

            # Update
            if self.is_moving:
                self.move_player()
            self.update_camera(dt)

            # Draw
            self.screen.fill(BACKGROUND_COLOR)

            # Draw obstacles in float, but pygame.draw requires int
            for obstacle in self.obstacles:
                # Subtract camera offset (floats)
                draw_rect = pygame.FRect(
                    obstacle.x - self.camera_offset[0],
                    obstacle.y - self.camera_offset[1],
                    obstacle.width,
                    obstacle.height
                )
                # Minimal integer usage (round) for drawing
                pygame.draw.rect(
                    self.screen,
                    OBSTACLE_COLOR,
                    (round(draw_rect.x),
                     round(draw_rect.y),
                     round(draw_rect.width),
                     round(draw_rect.height))
                )

            # Draw path if it exists
            if len(self.current_path) >= 2:
                offset_path = []
                for p in self.current_path:
                    offset_x = p[0] - self.camera_offset[0]
                    offset_y = p[1] - self.camera_offset[1]
                    offset_path.append((offset_x, offset_y))

                # Convert to int for drawing lines
                offset_path_int = [(round(px), round(py)) for (px, py) in offset_path]
                pygame.draw.lines(self.screen, PATH_COLOR, False, offset_path_int, width=2)

            # Draw player as a circle (again, must convert to int for actual draw)
            draw_x = self.player_pos[0] - self.camera_offset[0]
            draw_y = self.player_pos[1] - self.camera_offset[1]
            pygame.draw.circle(
                self.screen,
                PLAYER_COLOR,
                (round(draw_x), round(draw_y)),
                round(PLAYER_SIZE / 2.0)
            )

            # Show message (if any)
            if self.message:
                font = pygame.font.SysFont(None, 24)
                text_surf = font.render(self.message, True, (255, 0, 0))
                self.screen.blit(text_surf, (10, 10))

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
