import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDesktopWidget, QGridLayout

import QuoridorRules


board_tile_buttons = QGridLayout()

class MyApp(QWidget):

    whose_turn = 0
    number_of_players = 0
    is_obstacle_mode = False
    #장애물을 두기 위해 임시로 저장해두는 장애물의 중앙좌표
    temporary_obstacle_middle = [-1,-1]

    def __init__(self):
        super().__init__()
        self.init_main_menu()

    def init_main_menu(self):
        btn1 = QPushButton('1인 플레이', self)
        btn1.setCheckable(True)

        btn2 = QPushButton('2인 플레이', self)
        btn2.clicked.connect(self.init_2players_game)

        board_tile_buttons.addWidget(btn1)
        board_tile_buttons.addWidget(btn2)

        self.setLayout(board_tile_buttons)
        self.setWindowTitle('Quoridor')
        self.setGeometry(800, 800, 800, 800)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(cp)
        self.move(qr.topLeft())
        self.show()
        print(f'main menu')

    def remove_main_menu(self):
        for i in reversed(range(board_tile_buttons.count())):
            board_tile_buttons.itemAt(i).widget().deleteLater()
        print(f'main menu removed')

    def init_2players_game(self):
        self.number_of_players = 2
        self.remove_main_menu()
        QuoridorRules.init_game()

        global tile_buttons

        tile_buttons = [[QPushButton(self) for col in range(QuoridorRules.number_of_tiles_in_side)] for row in range(QuoridorRules.number_of_tiles_in_side)]

        #보드타일 초기화
        #왼쪽 하단을 0,0으로 초기화하기 위해 인덱스 조작 이후 타일 꺼내오는 과정 전용 인덱스 조작함수 만들 필요 있음
        #tile_button = QPushButton(self)
        for index_x in range(QuoridorRules.number_of_tiles_in_side):
            for index_y in range(QuoridorRules.number_of_tiles_in_side):
                tile_buttons[index_y][16 - index_x] = QPushButton(self)
                board_tile_buttons.addWidget(tile_buttons[index_y][16 - index_x], index_x, index_y)
                # 버튼을 클릭시 좌표계로 변환하여 함수로 전달
                tile_buttons[index_y][16 - index_x].clicked.connect(lambda ld, idxx = index_x, idxy = index_y: self.click_tile(idxy, 16 - idxx))
                tile_buttons[index_y][16 - index_x].setText(index_y.__str__() + " , " + (16 - index_x).__str__())

                if index_x % 2 == 0 and index_y % 2 == 0:
                    tile_buttons[index_y][16 - index_x].setStyleSheet('QPushButton {background-color: #A3C1DA; color: light gray;}')

        # 장애물 모드 버튼
        global obstacle_mode_button
        obstacle_mode_button = QPushButton(self)
        obstacle_mode_button.setText("장애물 두기")
        board_tile_buttons.addWidget(obstacle_mode_button, 8, QuoridorRules.number_of_tiles_in_side + 1)
        obstacle_mode_button.clicked.connect(self.click_obstacle_mode_button)

        # 장애물 개수 표기
        global player1_obstacle_button
        player1_obstacle_button = QPushButton(self)
        player1_obstacle_button.setText("1p 남은 장애물 : 10")
        board_tile_buttons.addWidget(player1_obstacle_button, 9, QuoridorRules.number_of_tiles_in_side + 2)

        global player2_obstacle_button
        player2_obstacle_button = QPushButton(self)
        player2_obstacle_button.setText("2p 남은 장애물 : 10")
        board_tile_buttons.addWidget(player2_obstacle_button, 7, QuoridorRules.number_of_tiles_in_side + 2)

        # 턴 플레이어 표기
        global whose_turn_button
        whose_turn_button = QPushButton(self)
        whose_turn_button.setText("플레이어 1의 턴")
        board_tile_buttons.addWidget(whose_turn_button, 8, QuoridorRules.number_of_tiles_in_side + 2)
        """
        #상하좌우 이동용 버튼
        tile_button = QPushButton(self)
        tile_button.setText("위")
        tile_button.clicked.connect(lambda: self.click_upward_button(self.whose_turn))
        board_tile_buttons.addWidget(tile_button, QuoridorRules.number_of_tiles_in_side, QuoridorRules.number_of_tiles_in_side + 1)
        tile_button = QPushButton(self)
        tile_button.setText("아래")
        board_tile_buttons.addWidget(tile_button, QuoridorRules.number_of_tiles_in_side + 1, QuoridorRules.number_of_tiles_in_side + 1)
        tile_button = QPushButton(self)
        tile_button.setText("왼쪽")
        board_tile_buttons.addWidget(tile_button, QuoridorRules.number_of_tiles_in_side + 1, QuoridorRules.number_of_tiles_in_side)
        tile_button = QPushButton(self)
        tile_button.setText("오른쪽")
        board_tile_buttons.addWidget(tile_button, QuoridorRules.number_of_tiles_in_side + 1, QuoridorRules.number_of_tiles_in_side + 2)
        """

        #플레이어 초기화
        for player_index in range(2):
            #print(tile_buttons[QuoridorRules.players_coordinates[player_index][0]][QuoridorRules.players_coordinates[player_index][1]])
            #print(QuoridorRules.players_coordinates[player_index])
            tile_buttons[QuoridorRules.players_coordinates[player_index][0]][QuoridorRules.players_coordinates[player_index][1]].setText((player_index + 1).__str__() + "플레이어")

        self.draw_board()

    def draw_board(self):
        self.whose_turn = 0
        print(f'draw board')

    def click_tile(self, coord_x, coord_y):
        print(self.is_obstacle_mode)
        if not self.is_obstacle_mode:
            cur_pos_x = QuoridorRules.players_coordinates[self.whose_turn][QuoridorRules.axis_x]
            cur_pos_y = QuoridorRules.players_coordinates[self.whose_turn][QuoridorRules.axis_y]
            if cur_pos_x == coord_x:
                if cur_pos_y - coord_y == 2 or cur_pos_y - coord_y == -2:
                    delta_y = int((coord_y - cur_pos_y) / 2)
                    if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, 0, delta_y) and QuoridorRules.is_there_another_player_already(coord_x, coord_y) == False:
                        QuoridorRules.move_player(self.whose_turn, 0, delta_y * 2)
                        self.update_player_position_in_ui(self.whose_turn, cur_pos_x, cur_pos_y, 0, delta_y * 2)
                        return
            if cur_pos_y == coord_y:
                if cur_pos_x - coord_x == 2 or cur_pos_x - coord_x == -2:
                    delta_x = int((coord_x - cur_pos_x) / 2)
                    if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, delta_x, 0) and QuoridorRules.is_there_another_player_already(coord_x, coord_y) == False:
                        QuoridorRules.move_player(self.whose_turn, delta_x * 2, 0)
                        self.update_player_position_in_ui(self.whose_turn, cur_pos_x, cur_pos_y, delta_x * 2, 0)
                        return

            # 다른 플레이어를 건너뛸 경우
            if self.is_passable_another_player(coord_x, coord_y):
                delta_x = coord_x - cur_pos_x
                delta_y = coord_y - cur_pos_y
                QuoridorRules.move_player(self.whose_turn, delta_x, delta_y)
                self.update_player_position_in_ui(self.whose_turn, cur_pos_x, cur_pos_y, delta_x, delta_y)
                return
            print(f'you can not move to there')

        else:
            if self.temporary_obstacle_middle == [-1,-1]:
                if coord_x % 2 == 0 or coord_y % 2 == 0:
                    print(f'you can not put obstacles there')
                    return
                if QuoridorRules.obstacles_in_board[coord_x][coord_y]:
                    print(f'there is an obstacle already')
                    return
                self.temporary_obstacle_middle = [coord_x, coord_y]
                print(f'touch adjacent board tile to decide axis of obstacle')
            else:
                temp_x = self.temporary_obstacle_middle[QuoridorRules.axis_x]
                temp_y = self.temporary_obstacle_middle[QuoridorRules.axis_y]
                if temp_x == coord_x:
                    if temp_y - coord_y == 1 or temp_y - coord_y == -1:
                        if QuoridorRules.is_obstacle_puttable(self.whose_turn, self.temporary_obstacle_middle[QuoridorRules.axis_x], self.temporary_obstacle_middle[QuoridorRules.axis_y], QuoridorRules.axis_y):
                            if QuoridorRules.is_there_route_for_players(self.temporary_obstacle_middle[QuoridorRules.axis_x], self.temporary_obstacle_middle[QuoridorRules.axis_y], QuoridorRules.axis_y):
                                print(f'before put obstacle')
                                QuoridorRules.put_obstacle(self.whose_turn, temp_x, temp_y, QuoridorRules.axis_y)
                                self.update_ui_after_putting_obstacle(temp_x, temp_y, QuoridorRules.axis_y)
                            else:
                                self.click_obstacle_mode_button()
                                print(f'you must make at least one possible route for both players')

                        else:
                            print(f'there is an obstacle already')

                if temp_y == coord_y:
                    if temp_x - coord_x == 1 or temp_x - coord_x == -1:
                        if QuoridorRules.is_obstacle_puttable(self.whose_turn, self.temporary_obstacle_middle[QuoridorRules.axis_x], self.temporary_obstacle_middle[QuoridorRules.axis_y], QuoridorRules.axis_x):
                            if QuoridorRules.is_there_route_for_players(self.temporary_obstacle_middle[QuoridorRules.axis_x], self.temporary_obstacle_middle[QuoridorRules.axis_y], QuoridorRules.axis_x):
                                print(f'before put obstacle')
                                QuoridorRules.put_obstacle(self.whose_turn, temp_x, temp_y, QuoridorRules.axis_x)
                                self.update_ui_after_putting_obstacle(temp_x, temp_y, QuoridorRules.axis_x)
                            else:
                                self.click_obstacle_mode_button()
                                print(f'you must make at least one possible route for both players')
                        else:
                            print(f'there is an obstacle already')

    """
    def click_upward_button(self, whoseturn):
        pos_x = QuoridorRules.players_coordinates[whoseturn][0]
        pos_y = QuoridorRules.players_coordinates[whoseturn][1]
        print(pos_x)
        print(pos_y)
        if QuoridorRules.is_player_moveable(pos_x, pos_y, 0, 1):
            QuoridorRules.move_player(whoseturn, 0, 2)
            self.update_player_position_in_ui(whoseturn, pos_x, pos_y, 0, 2)
        else:
            print(f'you can not go upward')
    """

    def is_passable_another_player(self, coord_x, coord_y):
        # 다른 플레이어를 건너뛸 수 있는지 체크
        cur_pos_x = QuoridorRules.players_coordinates[self.whose_turn][QuoridorRules.axis_x]
        cur_pos_y = QuoridorRules.players_coordinates[self.whose_turn][QuoridorRules.axis_y]
        delta_x = coord_x - cur_pos_x
        delta_y = coord_y - cur_pos_y

        if abs(delta_x) + abs(delta_y) != 4:
            return False

        if abs(delta_y) == 4:
            if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, 0, int(delta_y / 4)):
                if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y + int(delta_y / 2), 0, int(delta_y / 4)):
                    if QuoridorRules.is_there_another_player_already(cur_pos_x, cur_pos_y + int(delta_y / 2)):
                        if not QuoridorRules.is_there_another_player_already(coord_x, coord_y):
                            return True

        if abs(delta_x) == 4:
            if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, int(delta_y / 4), 0):
                if QuoridorRules.is_player_moveable(cur_pos_x + int(delta_x / 2), cur_pos_y, int(delta_x / 4), 0):
                    if QuoridorRules.is_there_another_player_already(cur_pos_x + int(delta_x / 2), cur_pos_y):
                        if not QuoridorRules.is_there_another_player_already(coord_x, coord_y):
                            return True

        if abs(delta_x) == 2 and abs(delta_y) == 2:
            if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, 0, int(delta_y / 2)):
                if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y + delta_y, int(delta_x / 2), 0):
                    if QuoridorRules.is_there_another_player_already(cur_pos_x, cur_pos_y + delta_y):
                        if not QuoridorRules.is_there_another_player_already(coord_x, coord_y):
                            return True
            if QuoridorRules.is_player_moveable(cur_pos_x, cur_pos_y, int(delta_x / 2), 0):
                if QuoridorRules.is_player_moveable(cur_pos_x + delta_x, cur_pos_y, 0, int(delta_y / 2)):
                    if QuoridorRules.is_there_another_player_already(cur_pos_x + delta_x, cur_pos_y):
                        if not QuoridorRules.is_there_another_player_already(coord_x, coord_y):
                            return True
        return False

    def update_player_position_in_ui(self, player_index, pos_x, pos_y, delta_x, delta_y):
        tile_button = self.get_tile_position_with_coordinates(pos_x, pos_y)
        tile_button.setText(" ")
        tile_button = self.get_tile_position_with_coordinates(pos_x + delta_x, pos_y + delta_y)
        tile_button.setText((player_index + 1).__str__() + "플레이어")
        self.change_turn()

    def update_ui_after_putting_obstacle(self, coord_x, coord_y, axis):
        tile_buttons[coord_x][coord_y].setStyleSheet('QPushButton {background-color: black; color: red;}')
        if axis == QuoridorRules.axis_x:
            tile_buttons[coord_x + 1][coord_y].setStyleSheet('QPushButton {background-color: black; color: red;}')
            tile_buttons[coord_x - 1][coord_y].setStyleSheet('QPushButton {background-color: black; color: red;}')

        if axis == QuoridorRules.axis_y:
            tile_buttons[coord_x][coord_y + 1].setStyleSheet('QPushButton {background-color: black; color: red;}')
            tile_buttons[coord_x][coord_y - 1].setStyleSheet('QPushButton {background-color: black; color: red;}')

        player1_obstacle_button.setText("1p 남은 장애물 : " + QuoridorRules.players_obstacles[0].__str__())
        player2_obstacle_button.setText("2p 남은 장애물 : " + QuoridorRules.players_obstacles[1].__str__())
        self.temporary_obstacle_middle = [-1, -1]
        self.is_obstacle_mode = False
        self.change_turn()

    #좌표계로 실제 UI에 그려지는 타일의 주소로 변환
    def get_tile_position_with_coordinates(self, coord_x, coord_y):
        return tile_buttons[coord_x][coord_y]

    def click_obstacle_mode_button(self):
        if QuoridorRules.players_obstacles[self.whose_turn] < 1:
            print(f'you do not have enough obstacle to put')
            return
        self.is_obstacle_mode = not self.is_obstacle_mode
        self.temporary_obstacle_middle = [-1, -1]
        obstacle_mode_button.setText("장애물 두기 모드 " + self.is_obstacle_mode.__str__())
        print(f'장애물 두기 모드 ' + self.is_obstacle_mode.__str__())

    def change_turn(self):
        self.whose_turn = 1 - self.whose_turn
        whose_turn_button.setText("플레이어 " + (self.whose_turn + 1).__str__() + " 의 턴")
        obstacle_mode_button.setText("장애물 두기 모드 " + self.is_obstacle_mode.__str__())

    def initUI(self):
        btn1 = QPushButton('&Button1', self)
        btn1.setCheckable(True)
        btn1.toggle()

        btn2 = QPushButton(self)
        btn2.setText('Button&2')

        btn3 = QPushButton('Button3', self)
        btn3.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        vbox.addWidget(btn3)

        self.setLayout(vbox)
        self.setWindowTitle('QPushButton')
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    #QuoridorRules.init_game()
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())