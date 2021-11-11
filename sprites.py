import pygame
import os
import config
import time


# ------------------------- GENERAL PURPOSE FUNCTIONS ------------------------- #
def check_bounds(row, col, game_map: list):  # used by dfs, bbs, a*
    options = []
    if col - 1 >= 0:
        options.append([row, col - 1])
    if row + 1 < len(game_map):
        options.append([row + 1, col])
    if col + 1 < len(game_map[row]):
        options.append([row, col + 1])
    if row - 1 >= 0:
        options.append([row - 1, col])
    return options


def list_refresh(current: list, update: list):  # used by bbs, a*
    len1, len2 = len(update), len(current)
    i, j = 0, 0
    while i < len1:
        cost, last_node = update[i][1][0], update[i][1][1]
        while j < len2:
            if last_node == current[j][1][1]:
                if cost < current[j][1][0]:
                    current.pop(j)
                    len2 -= 1
                    j -= 1
                else:
                    # if partial path to be added, has >= cost than partial path with same last node skip it
                    update.pop(i)
                    i -= 1
                    len1 -= 1
                    break
            j += 1
        j = 0
        i += 1
    current.extend(update)

    return current


# ------------------------- DFS FUNCTIONS ------------------------- #
def choose_best(options: list, game_map: list) -> list:  # used by dfs
    options.sort(key=lambda elem: game_map[elem[0]][elem[1]].cost(), reverse=True)
    return options


def compute_possible_jump(row, col, game_map: list, path: list):  # used by dfs
    options = check_bounds(row, col, game_map)
    tmp = [elem for elem in options if game_map[elem[0]][elem[1]] not in path]
    return tmp


def check_if_neighbours(a: list, b: list):  # used by dfs
    if (abs(a[0] - b[0]) == 1 and a[1] == b[1]) or (abs(a[1] - b[1]) == 1 and a[0] == b[0]):
        return True
    else:
        return False


# ------------------------- BFS FUNCTIONS ------------------------- #
def compute_traversal_score(target: list, game_map: list, position: list) -> int:  # used by bfs
    tmp = check_bounds(target[0], target[1], game_map)
    options = [elem for elem in tmp if elem != position]
    result = 0
    for opt in options:
        result += game_map[opt[0]][opt[1]].cost()
    return result / len(options)


def check_traversal(row, col, game_map: list) -> list:  # used by bfs
    options = []
    if row - 1 >= 0:
        options.append([row - 1, col])
    if col + 1 < len(game_map[row]):
        options.append([row, col + 1])
    if row + 1 < len(game_map):
        options.append([row + 1, col])
    if col - 1 >= 0:
        options.append([row, col - 1])

    options.sort(key=lambda elem: compute_traversal_score(elem, game_map, [row, col]))
    return options


# ------------------------- BBS FUNCTIONS ------------------------- #
def check_bounds_bbs(partial_path: list, game_map: list, goal: list):  # used by bbs
    curr_path: list = list(partial_path[0])
    cost, last_node = partial_path[1][0], partial_path[1][1]

    node = curr_path[len(curr_path) - 1]
    row, col = node[0], node[1]
    possible_options = check_bounds(row, col, game_map)

    options = [elem for elem in possible_options if elem not in curr_path]
    new_paths = []
    if goal in options:
        new_paths.append([curr_path + [goal], [cost + game_map[goal[0]][goal[1]].cost()], goal])
        return new_paths, True
    for opt in options:
        new_paths.append([curr_path + [opt], [cost + game_map[opt[0]][opt[1]].cost(), opt]])
    return new_paths, False


# ------------------------- A* FUNCTIONS ------------------------- #
def check_costs(game_map):  # used by a*
    row_length = len(game_map)
    col_length = len(game_map[0])
    min = 9999999
    for i in range(row_length):
        for j in range(col_length):
            cost = game_map[i][j].cost()
            if min > cost: min = cost
    return min


def a_star_heuristics(candidate: list, goal: list, min_tile):  # used by a*
    if candidate == goal:
        return 0
    edge1, edge2 = abs(candidate[0] - goal[0]) + 1, abs(candidate[1] - goal[1]) + 1

    manhattan_distance = int(edge1) + int(edge2) - 2

    approx_path_cost = manhattan_distance * min_tile
    return approx_path_cost


def check_bounds_a_star(partial_path: list, game_map: list, goal: list, min_tile):  # used by a*
    curr_path: list = list(partial_path[0])
    cost, last_node = partial_path[1][0], partial_path[1][1]

    node = curr_path[len(curr_path) - 1]
    row, col = node[0], node[1]
    possible_options = check_bounds(row, col, game_map)

    options = [elem for elem in possible_options if elem not in curr_path]
    new_paths = []
    if goal in options:
        new_paths.append(
            [curr_path + [goal], [cost + game_map[goal[0]][goal[1]].cost(), goal,
                                  cost + game_map[goal[0]][goal[1]].cost() + a_star_heuristics(goal, goal, min_tile)]])
        return new_paths, True
    for opt in options:
        tmp = cost + game_map[opt[0]][opt[1]].cost()
        new_paths.append(
            [curr_path + [opt], [tmp, opt,
                                 tmp + a_star_heuristics(opt, goal, min_tile)]])
    return new_paths, False


class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, row, col, file_name, transparent_color=None):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (config.TILE_SIZE, config.TILE_SIZE))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (col * config.TILE_SIZE, row * config.TILE_SIZE)
        self.row = row
        self.col = col


class Agent(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Agent, self).__init__(row, col, file_name, config.DARK_GREEN)

    def move_towards(self, row, col):
        row = row - self.row
        col = col - self.col
        self.rect.x += col
        self.rect.y += row

    def place_to(self, row, col):
        self.row = row
        self.col = col
        self.rect.x = col * config.TILE_SIZE
        self.rect.y = row * config.TILE_SIZE

    # game_map - list of lists of elements of type Tile
    # goal - (row, col)
    # return value - list of elements of type Tile
    def get_agent_path(self, game_map, goal):
        pass


class ExampleAgent(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        row = self.row
        col = self.col
        while True:
            if row != goal[0]:
                row = row + 1 if row < goal[0] else row - 1
            elif col != goal[1]:
                col = col + 1 if col < goal[1] else col - 1
            else:
                break
            path.append(game_map[row][col])
        return path


class Aki(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        dfs_list = []
        row = self.row
        col = self.col

        start = time.time()

        while True:
            options = compute_possible_jump(row, col, game_map, path)
            dfs_list.extend(choose_best(options, game_map))
            row, col = dfs_list.pop()

            if len(options) == 0:
                while True:
                    if not check_if_neighbours(list(path[-1].position()), [row, col]):  # backtracking if stuck
                        path.pop()
                    else:
                        break

            if row == goal[0] and col == goal[1]:
                path.append(game_map[row][col])
                break

            path.append(game_map[row][col])

        end = time.time()
        print(end - start)

        return path


class Jocke(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = []
        bfs_graph = {}
        curr_pop_list = []
        cnt = 0

        row = self.row
        col = self.col
        start = game_map[row][col]

        start_time = time.time()

        while True:
            options = check_traversal(row, col, game_map)
            bfs_graph[(row, col)] = options

            if [goal[0], goal[1]] in options:
                path.append(game_map[goal[0]][goal[1]])
                break

            if len(bfs_graph) == 1:  # starting state, get first list to process
                iterator = next(iter(bfs_graph))
                cnt = 1
                curr_pop_list = bfs_graph.get(iterator)

            if len(curr_pop_list) == 0:  # when curr list empty, find next to process
                iterator = iter(bfs_graph)
                tmp = ()
                for i in range(cnt):
                    next(iterator)
                tmp = next(iterator)
                cnt += 1  # increase position in dictionary for next list
                curr_pop_list = list(bfs_graph.get(tmp))

            row, col = curr_pop_list.pop(0)

        keys = list(bfs_graph.keys())
        values = list(bfs_graph.values())
        values_len = len(values)
        curr = [goal[0], goal[1]]

        while start not in path:
            for el in range(values_len):
                if curr in values[el]:
                    index = values.index(values[el])
                    values_len = el
                    break
            curr = list(keys[index])
            path.insert(0, game_map[curr[0]][curr[1]])

        end = time.time()
        print(end - start_time)

        return path


class Draza(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        row = self.row
        col = self.col

        path = []

        bbs_list = [[[[row, col]], [0, [row, col]]]]
        start = time.time()
        while True:
            partial_path = bbs_list.pop(0)
            new_paths, end = check_bounds_bbs(partial_path, game_map, list(goal))
            if not end:
                bbs_list = list_refresh(bbs_list, new_paths)
                bbs_list.sort(key=lambda elem: (elem[1][0], len(elem[0])))  # sort by price,
            else:  # then by num of nodes in partial path
                for elem in new_paths[0][0]:
                    path.append(game_map[elem[0]][elem[1]])
                break

        end = time.time()
        print(end - start)

        return path


class Bole(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        row = self.row
        col = self.col

        path = []
        a_star_list = [[[[row, col]], [0, [row, col], 0]]]
        min_tile = check_costs(game_map)

        start = time.time()
        while True:
            partial_path = a_star_list.pop(0)
            new_paths, end = check_bounds_a_star(partial_path, game_map, list(goal), min_tile)
            if not end:
                a_star_list = list_refresh(a_star_list, new_paths)
                a_star_list.sort(key=lambda elem: (
                    elem[1][2],
                    len(elem[0])))  # sort by sum of price and heuristic, then by num of nodes in partial path
            else:
                for elem in new_paths[0][0]:
                    path.append(game_map[elem[0]][elem[1]])
                break

        end = time.time()
        print(end - start)

        return path


class Tile(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Tile, self).__init__(row, col, file_name)

    def position(self):
        return self.row, self.col

    def cost(self):
        pass

    def kind(self):
        pass


class Stone(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'stone.png')

    def cost(self):
        return 1000

    def kind(self):
        return 's'


class Water(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'water.png')

    def cost(self):
        return 500

    def kind(self):
        return 'w'


class Road(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'road.png')

    def cost(self):
        return 2

    def kind(self):
        return 'r'


class Grass(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'grass.png')

    def cost(self):
        return 3

    def kind(self):
        return 'g'


class Mud(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'mud.png')

    def cost(self):
        return 5

    def kind(self):
        return 'm'


class Dune(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'dune.png')

    def cost(self):
        return 7

    def kind(self):
        return 's'


class Goal(BaseSprite):
    def __init__(self, row, col):
        super().__init__(row, col, 'x.png', config.DARK_GREEN)


class Trail(BaseSprite):
    def __init__(self, row, col, num):
        super().__init__(row, col, 'trail.png', config.DARK_GREEN)
        self.num = num

    def draw(self, screen):
        text = config.GAME_FONT.render(f'{self.num}', True, config.WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
