# 장애물을 따로 관리하는 경우
# 장애물을 둘 때 경로를 최소 하나 남겨놔야 하는 규칙이 지켜지는지
# 파악하는 과정에서 놓여있는 장애물의 수 n * 알고리즘을 통해 경로를 찾는
# 타일 수의 값만큼 연산이 필요
# 반면 장애물을 타일처럼 인덱스를 붙여두고 장애물이 놓여있는 곳을 bool값을
# 통해 관리하면 연산비용은 거의 필요없으나 추가 메모리가 4배 이상 필요

# 일단은 메모리를 써서 연산비용을 줄이는 방향으로
# 왼쪽아래 (0,0) 기준으로 타일의 인덱스는 플레이어가 움직일 수 있는 타일은 (짝수, 짝수)
# 장애물의 중심은 (홀수, 홀수) 장애물이 막는 통로는 (홀수, 짝수) 혹은 (짝수, 홀수)

# 장애물이 놓이 공간을 포함한 보드 한 면의 타일 수
number_of_tiles_in_side = 17

# x, y축에 대한 enum값
axis_x = 0
axis_y = 1

# 플레이어의 enum값
player1 = 0
player2 = 1

# 두 플레이어의 좌표
players_coordinates = [[0 for col in range(2)] for row in range(2)]
# 두 플레이어가 사용가능한 장애물의 수
players_obstacles = [0, 0]
obstacles_in_board = [[False for col in range(number_of_tiles_in_side)] for row in range(number_of_tiles_in_side)]
game_set = False

# A스타용 임시 저장 변수
checked_tiles = [[0, 0], [0, 0]]
checking_tiles = [[0, 0], [0, 0]]
is_there_available_route = [False, False]


def init_game():
    players_coordinates[player1] = [int((number_of_tiles_in_side - 1) / 2), 0]
    players_coordinates[player2] = [int((number_of_tiles_in_side - 1) / 2), number_of_tiles_in_side - 1]
    players_obstacles[player1] = 10
    players_obstacles[player2] = 10
    #  del obstacles_in_board[:]
    for index_x in range(number_of_tiles_in_side):
        for index_y in range(number_of_tiles_in_side):
            obstacles_in_board[index_x][index_y] = False
            # print(obstacles_in_board[index_x][index_y]);
    print(f'InitCompleted')


def is_player_moveable(position_x, position_y, delta_x, delta_y):
    # 2칸이 아닌 1칸을 델타 단위로 삼아 이동 가능할 경우 2칸으로
    next_pos = (position_x + delta_x, position_y + delta_y)
    # print(next_pos)
    if next_pos[axis_x] < 0 or next_pos[axis_x] > number_of_tiles_in_side - 1:
        return False
    elif next_pos[axis_y] < 0 or next_pos[axis_y] > number_of_tiles_in_side - 1:
        return False
    elif obstacles_in_board[next_pos[axis_x]][next_pos[axis_y]]:
        return False
    return True


def is_there_another_player_already(target_pos_x, target_pos_y):
    # 이동하려는 좌표에 다른 캐릭터가 있는지 체크
    if [target_pos_x, target_pos_y] in players_coordinates:
        return True
    return False


def move_player(player_index, delta_x, delta_y):
    # 여기서는 실제 이동이므로 델타값은 2칸씩
    # 다른 플레이어를 넘어가는 경우도 일괄처리
    players_coordinates[player_index] = [players_coordinates[player_index][axis_x] + delta_x, players_coordinates[player_index][axis_y] + delta_y]
    print((player_index + 1).__str__() + f' player moved to' + players_coordinates[player_index].__str__())
    is_game_set(player_index)


def is_obstacle_puttable(player_index, position_x, position_y, direction):
    # 일단 direction은 0이면 가로 1이면 세로
    # 보드게임 상 물리적으로 장애물을 둘 수 있는지 체크
    if players_obstacles[player_index] < 1:
        return False
    elif position_x * position_y % 2 == 0:
        return False
    elif obstacles_in_board[position_x][position_y]:
        return False
    elif direction == axis_x:
        if obstacles_in_board[position_x - 1][position_y] or obstacles_in_board[position_x + 1][position_y]:
            return False
    elif direction == axis_y:
        if obstacles_in_board[position_x][position_y - 1] or obstacles_in_board[position_x][position_y + 1]:
            return False
    #if is_there_route_for_players() == False:
        #return False
    return True


def is_there_route_for_players(pos_x, pos_y, axis):
    # 일단은 A스타로 짜긴 해야할것 같은데 너무 자원을 많이 먹을 것 같음.
    # 뭔가 획기적인 방법을 생각안하면 규칙 구현하는 것 때문에 프로젝트가 막힐듯.
    # 일단은 A스타로
    # 물리적 체크 후 규칙 상 하나 이상의 경로가 남는지 체크
    player_index = 0
    for players_coordinate in players_coordinates:
        #reachable_tiles = [[False] * number_of_tiles_in_side] * number_of_tiles_in_side
        #reachable_tiles[players_coordinate[axis_x]][players_coordinate[axis_y]] = True

        obstacle_middle_pos = [pos_x, pos_y]
        obstacle_side1_pos = [pos_x, pos_y]
        obstacle_side2_pos = [pos_x, pos_y]

        if axis == axis_x:
            obstacle_side1_pos = [pos_x- 1, pos_y]
            obstacle_side2_pos = [pos_x + 1, pos_y]
        elif axis == axis_y:
            obstacle_side1_pos = [pos_x, pos_y - 1]
            obstacle_side2_pos = [pos_x, pos_y + 1]

        obstacles_in_board[obstacle_middle_pos[axis_x]][obstacle_middle_pos[axis_y]] = True
        obstacles_in_board[obstacle_side1_pos[axis_x]][obstacle_side1_pos[axis_y]] = True
        obstacles_in_board[obstacle_side2_pos[axis_x]][obstacle_side2_pos[axis_y]] = True

        is_there_available_route[player_index] = False

        checked_tiles.clear()
        #checked_tiles.append(players_coordinate)

        checking_tiles.clear()
        checking_tiles.append(players_coordinate)
        recur_check_tile(player_index, players_coordinate[axis_x], players_coordinate[axis_y])

        obstacles_in_board[obstacle_middle_pos[axis_x]][obstacle_middle_pos[axis_y]] = False
        obstacles_in_board[obstacle_side1_pos[axis_x]][obstacle_side1_pos[axis_y]] = False
        obstacles_in_board[obstacle_side2_pos[axis_x]][obstacle_side2_pos[axis_y]] = False

        player_index = player_index + 1
    if is_there_available_route[player1] and is_there_available_route[player2]:
        return True
    return False


def recur_check_tile(player_index, position_x, position_y):
    print(f'recur_check_tile')
    if is_player_moveable(position_x, position_y, 0, 1):
        if player_index == player1 and position_y + 2 == number_of_tiles_in_side - 1:
            print(f'there is a route for player 1')
            is_there_available_route[player1] = True
            return
        if [position_x, position_y + 2] not in checked_tiles and [position_x, position_y + 2] not in checking_tiles:
            checking_tiles.append([position_x, position_y + 2])
    if is_player_moveable(position_x, position_y, 0, -1):
        if player_index == player2 and position_y - 2 == 0:
            print(f'there is a route for player 2')
            is_there_available_route[player2] = True
            return
        if [position_x, position_y - 2] not in checked_tiles and [position_x, position_y - 2] not in checking_tiles:
            checking_tiles.append([position_x, position_y - 2])
    if is_player_moveable(position_x, position_y, 1, 0):
        if [position_x + 2, position_y] not in checked_tiles and [position_x + 2, position_y] not in checking_tiles:
            checking_tiles.append([position_x + 2, position_y])
        #if checked_tiles.__contains__([position_x + 2, position_y]) == False and checking_tiles.__contains__([position_x + 2, position_y]) == False:
    if is_player_moveable(position_x, position_y, -1, 0):
        if [position_x - 2, position_y] not in checked_tiles and [position_x - 2, position_y] not in checking_tiles:
            checking_tiles.append([position_x - 2, position_y])
    checking_tiles.remove([position_x, position_y])
    checked_tiles.append([position_x, position_y])
    print(len(checked_tiles).__str__() + f'checked_tiles')
    print(len(checking_tiles).__str__() + f'checking_tiles')
    if len(checking_tiles) != 0:
        coord_x = checking_tiles[len(checking_tiles) - 1][axis_x]
        coord_y = checking_tiles[len(checking_tiles) - 1][axis_y]
        recur_check_tile(player_index, coord_x, coord_y)


def put_obstacle(player_index, position_x, position_y, direction):
    # 장애물을 둘 수 있는지 확인 한 후에 실행된다고 가정
    if players_obstacles[player_index] > 0:
        players_obstacles[player_index] = players_obstacles[player_index] - 1
        obstacles_in_board[position_x][position_y] = True
        if direction == axis_x:
            obstacles_in_board[position_x - 1][position_y] = True
            obstacles_in_board[position_x + 1][position_y] = True
        if direction == axis_y:
            obstacles_in_board[position_x][position_y - 1] = True
            obstacles_in_board[position_x][position_y + 1] = True
        print((player_index + 1).__str__() + f' put obstacle at ' + position_x.__str__() + position_y.__str__() + direction.__str__())


def is_game_set(player_index):
    if player_index == player1:
        if players_coordinates[player_index][axis_y] == number_of_tiles_in_side - 1:
            print(f'player1 won')
            return True
    elif player_index == player2:
        if players_coordinates[player_index][axis_y] == 0:
            print(f'player2 won')
            return True
    return False