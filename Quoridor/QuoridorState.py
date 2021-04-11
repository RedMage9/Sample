# 장애물을 따로 관리하는 경우
# 장애물을 둘 때 경로를 최소 하나 남겨놔야 하는 규칙이 지켜지는지
# 파악하는 과정에서 놓여있는 장애물의 수 n * 알고리즘을 통해 경로를 찾는
# 타일 수의 값만큼 연산이 필요
# 반면 장애물을 타일처럼 인덱스를 붙여두고 장애물이 놓여있는 곳을 int값을
# 통해 관리하면 연산비용은 거의 필요없으나 추가 메모리가 4배 이상 필요

# 일단은 메모리를 써서 연산비용을 줄이는 방향으로
# 왼쪽아래 (0,0) 기준으로 타일의 인덱스는 플레이어가 움직일 수 있는 타일은 (짝수, 짝수)
# 장애물의 중심은 (홀수, 홀수) 장애물이 막는 통로는 (홀수, 짝수) 혹은 (짝수, 홀수)

import numpy as np


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
obstacles_in_board = [[0.0 for col in range(number_of_tiles_in_side)] for row in range(number_of_tiles_in_side)]
game_set = False

# A스타용 임시 저장 변수
checked_tiles = [[0, 0], [0, 0]]
checking_tiles = [[0, 0], [0, 0]]
is_there_available_route = [False, False]

# 게임 진행 세부 변수
whose_turn = 0
winner_index = -1
number_of_players = 0
available_act = []

# 게임 초기화 및 턴 관련 함수


def init_game(number_of_players_param):
    # 2명 초과일 경우 코드 추가 필요
    players_coordinates[player1] = [8, 0]
    players_coordinates[player2] = [8, number_of_tiles_in_side - 1]
    players_obstacles[player1] = 5
    players_obstacles[player2] = 5
    for index_x in range(number_of_tiles_in_side):
        for index_y in range(number_of_tiles_in_side):
            obstacles_in_board[index_x][index_y] = 0.0
    global whose_turn
    whose_turn = 0

    global winner_index
    winner_index = -1

    global number_of_players
    number_of_players = number_of_players_param

    global available_act
    available_act = np.array(range(140))

    #print(f'Init Completed')


def change_turn():
    global whose_turn
    whose_turn = get_next_players_index(whose_turn)
    # print((whose_turn + 1).__str__() + f' players turn ')


def is_game_set():
    # 플레이어 승리 프린트 다른곳으로 옮길 필요 있음
    #if whose_turn == player1:
    global winner_index

    if players_coordinates[player1][axis_y] == number_of_tiles_in_side - 1:
        winner_index = player1
            #print(f'player1 won')
        return True
    #elif whose_turn == player2:
    if players_coordinates[player2][axis_y] == 0:
        winner_index = player2
            #print(f'player2 won')
        return True
    return False

# -------------------------------------


# -------------------------------------
# 이동 관련 함수


def is_tile_movable_to(coord_x, coord_y):
    cur_pos_x = players_coordinates[whose_turn][axis_x]
    cur_pos_y = players_coordinates[whose_turn][axis_y]
    if cur_pos_x == coord_x:
        if cur_pos_y - coord_y == 2 or cur_pos_y - coord_y == -2:
            delta_y = int((coord_y - cur_pos_y) / 2)
            if is_player_movable(cur_pos_x, cur_pos_y, 0, delta_y) and is_there_another_player_already(coord_x, coord_y) == False:
                return True
    if cur_pos_y == coord_y:
        if cur_pos_x - coord_x == 2 or cur_pos_x - coord_x == -2:
            delta_x = int((coord_x - cur_pos_x) / 2)
            if is_player_movable(cur_pos_x, cur_pos_y, delta_x, 0) and is_there_another_player_already(coord_x, coord_y) == False:
                return True

    # 다른 플레이어를 건너뛸 경우
    if is_passable_another_player(coord_x, coord_y):
        delta_x = coord_x - cur_pos_x
        delta_y = coord_y - cur_pos_y
        return True
    #print(f'you can not move to there')
    return False


def is_player_movable(position_x, position_y, delta_x, delta_y):
    # 2칸이 아닌 1칸을 델타 단위로 삼아 이동 가능할 경우 2칸으로
    next_pos = (position_x + delta_x, position_y + delta_y)
    # print(next_pos)
    if next_pos[axis_x] < 0 or next_pos[axis_x] > number_of_tiles_in_side - 1:
        return False
    elif next_pos[axis_y] < 0 or next_pos[axis_y] > number_of_tiles_in_side - 1:
        return False
    elif obstacles_in_board[next_pos[axis_x]][next_pos[axis_y]] == 1.0:
        return False
    return True


def is_passable_another_player(coord_x, coord_y):
    # 다른 플레이어를 건너뛸 수 있는지 체크
    cur_pos_x = players_coordinates[whose_turn][axis_x]
    cur_pos_y = players_coordinates[whose_turn][axis_y]
    delta_x = coord_x - cur_pos_x
    delta_y = coord_y - cur_pos_y

    if abs(delta_x) + abs(delta_y) != 4:
        return False

    if abs(delta_y) == 4:
        if is_player_movable(cur_pos_x, cur_pos_y, 0, int(delta_y / 4)):
            if is_player_movable(cur_pos_x, cur_pos_y + int(delta_y / 2), 0, int(delta_y / 4)):
                if is_there_another_player_already(cur_pos_x, cur_pos_y + int(delta_y / 2)):
                    if not is_there_another_player_already(coord_x, coord_y):
                        return True

    if abs(delta_x) == 4:
        if is_player_movable(cur_pos_x, cur_pos_y, int(delta_y / 4), 0):
            if is_player_movable(cur_pos_x + int(delta_x / 2), cur_pos_y, int(delta_x / 4), 0):
                if is_there_another_player_already(cur_pos_x + int(delta_x / 2), cur_pos_y):
                    if not is_there_another_player_already(coord_x, coord_y):
                        return True

    if abs(delta_x) == 2 and abs(delta_y) == 2:
        if is_player_movable(cur_pos_x, cur_pos_y, 0, int(delta_y / 2)):
            if is_player_movable(cur_pos_x, cur_pos_y + delta_y, int(delta_x / 2), 0):
                if is_there_another_player_already(cur_pos_x, cur_pos_y + delta_y):
                    if not is_there_another_player_already(coord_x, coord_y):
                        return True

        if is_player_movable(cur_pos_x, cur_pos_y, int(delta_x / 2), 0):
            if is_player_movable(cur_pos_x + delta_x, cur_pos_y, 0, int(delta_y / 2)):
                if is_there_another_player_already(cur_pos_x + delta_x, cur_pos_y):
                    if not is_there_another_player_already(coord_x, coord_y):
                        return True
    return False


def is_there_another_player_already(target_pos_x, target_pos_y):
    # 이동하려는 좌표에 다른 캐릭터가 있는지 체크
    if [target_pos_x, target_pos_y] in players_coordinates:
        return True
    return False


def move_player(player_index, delta_x, delta_y):
    # 여기서는 실제 이동이므로 델타값은 2칸씩
    # 다른 플레이어를 넘어가는 경우도 일괄처리
    coord_for_printing = players_coordinates[player_index]
    players_coordinates[player_index] = [players_coordinates[player_index][axis_x] + delta_x, players_coordinates[player_index][axis_y] + delta_y]
    #print((player_index + 1).__str__() + f' player moved from ' + coord_for_printing.__str__()
    #      + f' to ' + players_coordinates[player_index].__str__())
    #is_game_set()

# -------------------------------------


# -------------------------------------
# 장애물 배치 관련 함수


def is_obstacle_puttable(player_index, position_x, position_y, direction):
    # 일단 direction은 0이면 가로 1이면 세로
    # 보드게임 상 물리적으로 장애물을 둘 수 있는지 체크
    if players_obstacles[player_index] < 1:
        return False
    elif position_x * position_y % 2 == 0:
        return False
    elif obstacles_in_board[position_x][position_y] == 1.0:
        return False
    elif direction == axis_x:
        if obstacles_in_board[position_x - 1][position_y] == 1.0 or obstacles_in_board[position_x + 1][position_y] == 1.0:
            return False
    elif direction == axis_y:
        if obstacles_in_board[position_x][position_y - 1] == 1.0 or obstacles_in_board[position_x][position_y + 1] == 1.0:
            return False
    return is_there_route_for_players(position_x, position_y, direction)


def is_there_route_for_players(pos_x, pos_y, axis):
    # 일단은 A스타로 짜긴 해야할것 같은데 너무 자원을 많이 먹을 것 같음.
    # 뭔가 획기적인 방법을 생각안하면 규칙 구현하는 것 때문에 프로젝트가 막힐듯.
    # 일단은 A스타로
    # 물리적 체크 후 규칙 상 하나 이상의 경로가 남는지 체크
    player_index = 0

    for players_coordinate in players_coordinates:
        obstacle_middle_pos = [pos_x, pos_y]
        obstacle_side1_pos = [pos_x, pos_y]
        obstacle_side2_pos = [pos_x, pos_y]

        if axis == axis_x:
            obstacle_side1_pos = [pos_x- 1, pos_y]
            obstacle_side2_pos = [pos_x + 1, pos_y]
        elif axis == axis_y:
            obstacle_side1_pos = [pos_x, pos_y - 1]
            obstacle_side2_pos = [pos_x, pos_y + 1]

        obstacles_in_board[obstacle_middle_pos[axis_x]][obstacle_middle_pos[axis_y]] = 1.0
        obstacles_in_board[obstacle_side1_pos[axis_x]][obstacle_side1_pos[axis_y]] = 1.0
        obstacles_in_board[obstacle_side2_pos[axis_x]][obstacle_side2_pos[axis_y]] = 1.0

        is_there_available_route[player_index] = False

        checked_tiles.clear()
        #checked_tiles.append(players_coordinate)

        checking_tiles.clear()
        checking_tiles.append(players_coordinate)
        recur_check_tile(player_index, players_coordinate[axis_x], players_coordinate[axis_y])

        obstacles_in_board[obstacle_middle_pos[axis_x]][obstacle_middle_pos[axis_y]] = 0.0
        obstacles_in_board[obstacle_side1_pos[axis_x]][obstacle_side1_pos[axis_y]] = 0.0
        obstacles_in_board[obstacle_side2_pos[axis_x]][obstacle_side2_pos[axis_y]] = 0.0

        player_index = player_index + 1

    if is_there_available_route[player1] and is_there_available_route[player2]:
        return True

    act_number = get_put_obstacle_number_by_indices(pos_x, pos_y, axis)
    set_unavailable_act(act_number)
    return False


def recur_check_tile(player_index, position_x, position_y):
    #print(f'recur_check_tile')
    if is_player_movable(position_x, position_y, 0, 1):
        if player_index == player1 and position_y + 2 == number_of_tiles_in_side - 1:
            #print(f'there is a route for player 1')
            is_there_available_route[player1] = True
            return
        if [position_x, position_y + 2] not in checked_tiles and [position_x, position_y + 2] not in checking_tiles:
            checking_tiles.append([position_x, position_y + 2])
    if is_player_movable(position_x, position_y, 0, -1):
        if player_index == player2 and position_y - 2 == 0:
            #print(f'there is a route for player 2')
            is_there_available_route[player2] = True
            return
        if [position_x, position_y - 2] not in checked_tiles and [position_x, position_y - 2] not in checking_tiles:
            checking_tiles.append([position_x, position_y - 2])
    if is_player_movable(position_x, position_y, 1, 0):
        if [position_x + 2, position_y] not in checked_tiles and [position_x + 2, position_y] not in checking_tiles:
            checking_tiles.append([position_x + 2, position_y])
        #if checked_tiles.__contains__([position_x + 2, position_y]) == False and checking_tiles.__contains__([position_x + 2, position_y]) == False:
    if is_player_movable(position_x, position_y, -1, 0):
        if [position_x - 2, position_y] not in checked_tiles and [position_x - 2, position_y] not in checking_tiles:
            checking_tiles.append([position_x - 2, position_y])
    checking_tiles.remove([position_x, position_y])
    checked_tiles.append([position_x, position_y])
    #print(len(checked_tiles).__str__() + f'checked_tiles')
    #print(len(checking_tiles).__str__() + f'checking_tiles')
    if len(checking_tiles) != 0:
        coord_x = checking_tiles[len(checking_tiles) - 1][axis_x]
        coord_y = checking_tiles[len(checking_tiles) - 1][axis_y]
        recur_check_tile(player_index, coord_x, coord_y)


def put_obstacle(player_index, position_x, position_y, direction):
    # 장애물을 둘 수 있는지 확인 한 후에 실행된다고 가정
    if players_obstacles[player_index] > 0:
        players_obstacles[player_index] = players_obstacles[player_index] - 1
        obstacles_in_board[position_x][position_y] = 1.0

        x_index = ((position_x - 1) / 2).__int__()
        y_index = ((position_y - 1) / 2).__int__()
        act_number = get_put_obstacle_number_by_indices(position_x, position_y, direction)
        set_unavailable_act(act_number)
        #act_number = get_put_obstacle_number_by_indices(position_x, position_y, 1 - direction)
        set_unavailable_act(act_number + 1 - 2 * direction)

        if direction == axis_x:
            obstacles_in_board[position_x - 1][position_y] = 1.0
            if x_index - 1 > -1:
                set_unavailable_act(((x_index - 1) * 8 + y_index) * 2 + 12 + direction)

            obstacles_in_board[position_x + 1][position_y] = 1.0
            if x_index + 1 < 8:
                set_unavailable_act(((x_index + 1) * 8 + y_index) * 2 + 12 + direction)

        if direction == axis_y:
            obstacles_in_board[position_x][position_y - 1] = 1.0
            if y_index - 1 > -1:
                set_unavailable_act((x_index * 8 + y_index - 1) * 2 + 12 + direction)

            obstacles_in_board[position_x][position_y + 1] = 1.0
            if y_index + 1 < 8:
                set_unavailable_act((x_index * 8 + y_index + 1) * 2 + 12 + direction)


        #print((player_index + 1).__str__() + f' player put obstacle at ' + [position_x, position_y].__str__()
        #      + direction.__str__())


# -------------------------------------



# -------------------------------------
# 인덱스를 용도에 따라 변환하는 함수

# 장애물의 위치를 바이너리로 리턴.
# agent의 연산에 쓰이므로 상대위치로 변환해 리턴
def get_obstacles_in_board_by_binary(player_index):

    binary_obstacles = [[0.0 for col in range(number_of_tiles_in_side+3)] for row in range(number_of_tiles_in_side+3)]

    for index_x in range(number_of_tiles_in_side):
        for index_y in range(number_of_tiles_in_side):
            if player_index == player1:
                binary_obstacles[index_x][index_y] = obstacles_in_board[index_x][index_y]
            elif player_index == player2:
                binary_obstacles[index_x][index_y] = obstacles_in_board[16 - index_x][16 - index_y]

    return binary_obstacles


def get_additional_feature_vector(feature_vector, player_index):
    # 20 * 20의 배열의 형태로 가공

    feature_vector[19][0] = players_coordinates[player_index][0]
    feature_vector[19][1] = players_coordinates[player_index][1] * (1 - player_index) \
                            + (16 - players_coordinates[player_index][1]) * player_index

    feature_vector[0][19] = players_coordinates[1 - player_index][0]
    feature_vector[1][19] = players_coordinates[1 - player_index][1] * (1 - player_index) \
                            + (16 - players_coordinates[1 - player_index][1]) * player_index

    #feature_vector[19][10] = players_obstacles[player_index]
    #feature_vector[10][19] = players_obstacles[1 - player_index]

    #feature_vector[18][18] = 1 - whose_turn
    #feature_vector[19][18] = 1 - whose_turn
    #feature_vector[18][19] = whose_turn
    feature_vector[19][19] = whose_turn

    #feature_vector[players_coordinates[player_index][0]][players_coordinates[player_index][1]] = 9.0 - 5 * player_index
    #feature_vector[players_coordinates[1 - player_index][0]][players_coordinates[1 - player_index][1]] = 4.0 + 5 * player_index

    # print(np.array([feature_vector]))

    return feature_vector


def get_next_players_index(cur_idx):
    return (cur_idx + 1) % number_of_players


def get_tile_index_by_move_player_index(move_player_index):
    # delta index 기준

    position_x, position_y = 0, 0

    if move_player_index < 8:

        digit3 = int(move_player_index / 4) + 1

        delta = move_player_index - (digit3 - 1) * 4
        digit1 = int(delta / 2)
        digit2 = delta - 2 * digit1
        #print(digit1)
        #print(digit2)


        position_x = digit3 * 2 * (digit1 + -1 * digit2)
        position_y = digit3 * 2 * ((1 - digit1) + -1 * digit2)

    elif move_player_index < 12:

        move_player_index -= 8

        if move_player_index == 0:
            position_x = 2
            position_y = 2

        elif move_player_index == 1:
            position_x = 2
            position_y = -2

        elif move_player_index == 2:
            position_x = -2
            position_y = 2

        elif move_player_index == 3:
            position_x = -2
            position_y = -2

    return position_x, position_y


def get_put_obstacle_number_by_indices(position_x, position_y, direction):
    x_index = ((position_x - 1) / 2).__int__()
    y_index = ((position_y - 1) / 2).__int__()
    return (x_index * 8 + y_index) * 2 + 12 + direction


def get_indices_by_put_obstacle_number(number):
    number -= 12
    direction = number % 2
    number = (number - direction) / 2
    y_index = (number % 8).__int__()
    x_index = ((number - y_index) / 8).__int__()
    position_y = y_index * 2 + 1
    position_x = x_index * 2 + 1

    """
    print(direction)
    print(position_y)
    print(position_x)
    """
    return position_x, position_y, direction


def set_unavailable_act(act_number):
    # 불가능한 행동에 대한 집합을 유지하는게 아니라
    # 가능한 행동에 대한 집합을 유지
    # 그러나 장애물을 둘 때 경로를 하나 이상 남겨야 한다는 규칙때문에
    # 이 집합에 행동 인덱스가 있더라도 A스타로 체크후 수정해줘야함
    if act_number < 12 or act_number > 139:
        return
    #if not unavailable_act.__contains__(act_number):
    #    unavailable_act.append(act_number)

    global available_act

    #print(available_act)
    #print(act_number)

    if available_act.__contains__(act_number):
        #available_act = np.delete(available_act, [act_number])
        available_act = get_relative_complement(available_act, [act_number])


def get_relative_complement(set_a, set_b):
    relative_complement = []

    for i in range(len(set_a)):
        if not set_b.__contains__(set_a[i]):
            relative_complement.append(set_a[i])

    return relative_complement


def convert_absolute_logits_index_by_relatives(logit_indices, player_index):

    if player_index == 0:
        return logit_indices

    absolute_logit_indices = []

    for i in range(len(logit_indices)):
        if player_index == 1:
            if logit_indices[i] < 4:
                absolute_logit_indices.append(3 - logit_indices[i])
            elif logit_indices[i] < 8:
                absolute_logit_indices.append(11 - logit_indices[i])
            elif logit_indices[i] < 12:
                absolute_logit_indices.append(19 - logit_indices[i])
            elif logit_indices[i] < 140:
                absolute_logit_indices.append(151 - logit_indices[i])

    absolute_logit_indices.sort()

    return absolute_logit_indices


def convert_absolute_logit_index_by_relative(logit_index, player_index):

    if player_index == 0:
        return logit_index

    if player_index == 1:
        if logit_index < 4:
            logit_index = 3 - logit_index
        elif logit_index < 8:
            logit_index = 11 - logit_index
        elif logit_index < 12:
            logit_index = 19 - logit_index
        elif logit_index < 140:
            logit_index = 151 - logit_index

    return logit_index


def print_abs_map_status():
    board = get_obstacles_in_board_by_binary(0)
    map = [[0.0 for col in range(number_of_tiles_in_side+3)] for row in range(number_of_tiles_in_side+3)]
    for i in range(20):
        for ii in range(20):
            map[19 - ii][i] = board[i][ii]

    for i in range(20):
        #print(map[i])
        print(map[i])