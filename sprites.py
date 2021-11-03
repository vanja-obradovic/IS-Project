import pygame
import os
import config
import pprint


def check_bounds(row, col, game_map: list):
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


def compute_possible_jump(row, col, game_map: list, path: list):
    options = check_bounds(row, col, game_map)
    tmp = [elem for elem in options if game_map[elem[0]][elem[1]] not in path]
    return tmp


def check_if_neighbours(a: list, b: list):
    if (abs(a[0] - b[0]) == 1 and a[1] == b[1]) or (abs(a[1] - b[1]) == 1 and a[0] == b[0]):
        return True
    else:
        return False


def compute_traversal_score(target: list, game_map: list, position: list) -> int:
    tmp = check_bounds(target[0], target[1], game_map)
    options = [elem for elem in tmp if elem != position]
    result = 0
    for opt in options:
        result += game_map[opt[0]][opt[1]].cost()
    return result / len(options)


def check_traversal(row, col, game_map: list, path: list) -> list:
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
    # print(options)
    # for el in options:
    #     print(compute_traversal_score(el, game_map, [row, col]))
    return options


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

    def __choose_best(self, options: list, game_map: list) -> list:
        options.sort(key=lambda elem: game_map[elem[0]][elem[1]].cost(), reverse=True)
        return options

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        dfs_list = []
        row = self.row
        col = self.col
        while True:
            options = compute_possible_jump(row, col, game_map, path)
            dfs_list.extend(Aki.__choose_best(self, options, game_map))
            row, col = dfs_list.pop()
            if len(options) == 0:
                while True:
                    if not check_if_neighbours(list(path[-1].position()), [row, col]):
                        path.pop()
                    else:
                        break
            if row == goal[0] and col == goal[1]:
                path.append(game_map[row][col])
                break
            path.append(game_map[row][col])
        return path


class Jocke(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = []
        row = self.row
        col = self.col
        start = game_map[row][col]
        # bfs_list = []
        bfs_graph = {}
        # print(check_traversal(row, col, game_map, path))
        curr_pop_list = []
        while True:
            options = check_traversal(row, col, game_map, path)
            bfs_graph[(row, col)] = options
            if len(bfs_graph) == 1:
                iterator = next(iter(bfs_graph))
                cnt = 1
                curr_pop_list = bfs_graph.get(iterator)
            if [goal[0], goal[1]] in options:
                path.append(game_map[goal[0]][goal[1]])
                break
            if len(curr_pop_list) == 0:
                iterator = iter(bfs_graph)
                tmp = ()
                for i in range(cnt):
                    next(iterator)
                tmp = next(iterator)
                cnt += 1
                curr_pop_list = list(bfs_graph.get(tmp))
            row, col = curr_pop_list.pop(0)

            # bfs_list.extend(options)
            # row, col = bfs_list.pop(0)

        keys = list(bfs_graph.keys())
        values = list(bfs_graph.values())
        values_len = len(values)
        curr = [goal[0], goal[1]]

        print("Dictionary:")
        pprint.pprint(bfs_graph.items())
        print("Starting path calculation...\n")
        while start not in path:
            for el in range(values_len):
                if curr in values[el]:
                    print(values[el])
                    index = values.index(values[el])
                    values_len = el
                    break
            curr = list(keys[index])
            print("Parent" + " " + str(curr))
            path.insert(0, game_map[curr[0]][curr[1]])
        return path


class Draza(Agent):
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


class Bole(Agent):
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