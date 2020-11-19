# 비동기로 작동하는 프로세스의 객체개념

import numpy as np
import QuoridorState
import CNN
import tensorflow as tf
import random
from tensorflow.keras import datasets, layers, models


class Agent:

    # 학습절차의 설정에 관한 패러미터

    player_index = -1
    process_number = -1

    # 이후 학습을 위해 행동시 캐싱해둔 입력값
    prior_act = 0
    probability_for_prior_act = 1.0
    cashed_input = tf.reshape([[0.0 for col in range(CNN.number_of_input_side)] for row in range(CNN.number_of_input_side)], [-1, CNN.number_of_input_side, CNN.number_of_input_side, 1])
    cashed_feature_vector = np.array([])

    # -------------------------------------

    # 학습 패러미터
    # theta_v는 cnn쪽에서 공통관리
    #theta_pi = np.array([])
    #theta_v = np.array([])

    thread_step_counter = 1

    # -------------------------------------

    # Input Parameters
    # 논문을 통해 적절 수치 확인 필요
    alpha_theta_pi = 0.01
    alpha_theta_v = 0.01

    lambda_theta_pi = 0.99
    lambda_theta_v = 0.8

    discount_factor = 0.99

    moving_average_parameter = 0.9
    initial_learning_rate = 0.01
    epsilon = 1e-10

    t_max = 10

    # -------------------------------------

    # Local Parameters
    # 학습과정에서 단기로 사용하는 패러미터

    dutch_trace_theta_pi = np.array([])
    dutch_trace_theta_v = np.array([])
    #value_func = 0.0
    value_func_old = 0.0
    i_discount_factor = 1.0

    # phi는 특징벡터이므로 256 + 7개
    #phi = np.array([])
    #phi_next = np.array([])

    # -------------------------------------

    def __init__(self, theta_global, theta_v_global, g_pi_global, g_v_global, process_number):

        # 클래스에 선언해도 각 객체가 동일한 주솟값을 캐싱하게 됨
        # 여기에 선언하니 주솟값이 달라짐

        self.get_feature_vector_and_value_func = CNN.create_feature_vector_and_value_func_model()
        #self.direct_value_func = CNN.direct_value_func_model()

        #self.value_func_opt = tf.keras.optimizers.RMSprop(learning_rate=self.alpha_theta_v)
        self.value_func_opt = tf.keras.optimizers.SGD(learning_rate=self.alpha_theta_v)

        self.get_logit = CNN.create_logit_model()


        # 전역변수에 2차원 벡터를 지정하니 업데이트할때 메모리 가비지가 생겨서
        # 인덱스를 조정해 1차원벡터에 집어넣는 방식으로 수정
        # 동기화시 한쪽은 eager tensor고 한쪽은 1차원벡터라 2번 경유해야함

        self.Async_theta_pi_by_theta_pi_global(theta_global, theta_v_global, g_pi_global, g_v_global)

        self.process_number = process_number

        self.acted_before = False

        return

    def init_for_episode(self, player_index):
        self.dutch_trace_theta_pi = 0.0
        self.dutch_trace_theta_v = 0.0
        self.value_func_old = 0.0
        self.i_discount_factor = 1.0
        self.player_index = player_index

        self.prior_act = 0
        self.probability_for_prior_act = 1.0
        self.cashed_input = tf.reshape([[0.0 for col in range(CNN.number_of_input_side)] for row in range(CNN.number_of_input_side)], [-1, CNN.number_of_input_side, CNN.number_of_input_side, 1])
        self.cashed_feature_vector = tf.reshape([0.0 for col in range(CNN.number_of_feature_vector)], [-1, CNN.number_of_feature_vector, 1])
        self.acted_before = False

        return

    def act_by_policy(self):
        # 현재 에이전트의 사후 학습을 위해 현재 에이전트의 input을 캐싱
        # CNN 소프트맥스를 기반으로 행동하는 함수
        # softmax 구조상 불가능한 행동에 대해서도 logit이 존재해야하므로 전체 logit을 리턴받고
        # 실행 불가능한 행동을 제외한 후 실행
        # 0~3은 1칸 이동, 4~11은 다른 플레이어를 건너뛰어 2칸 이동
        # 상대플레이어를 뛰어넘는 것을 포함한 이동의 수 12 + 행동의 가짓수는 장애물을 두는 위치 64칸 * 가로세로 2로 140가지

        # 저번 턴에 학습할 때 사용한 input vector를 캐싱해두면 다시 구할 필요가 없을듯
        self.cashed_input = self.get_input_vector()
        #print(self.cashed_input)

        CNN_outputs = self.get_feature_vector_and_value_func(self.cashed_input, training=False)
        self.cashed_feature_vector = tf.reshape(np.transpose(CNN_outputs[0][0]), [-1, CNN.number_of_feature_vector, 1])
        #self.cashed_feature_vector = self.cashed_input

        # logit은 140개
        logits = self.get_logit(self.cashed_feature_vector, training=False)[0]

        # 불가능한 행동 집합은 logit및 softmax에 대한 인덱스
        index_of_available_act = QuoridorState.available_act
        unavailable_act = []

        for i in range(len(index_of_available_act)):
            if index_of_available_act[i] < 12:
                # 이동 인덱스로 바꾸는 함수
                position_x, position_y = QuoridorState.get_tile_index_by_move_player_index(index_of_available_act[i])
                position_x = QuoridorState.players_coordinates[QuoridorState.whose_turn][0] + position_x
                position_y = QuoridorState.players_coordinates[QuoridorState.whose_turn][1] + position_y
                if not QuoridorState.is_tile_movable_to(position_x, position_y):
                    unavailable_act.append(index_of_available_act[i])

            if index_of_available_act[i] > 11:
                # 장애물 인덱스로 바꾸는 함수
                position_x, position_y, direction = QuoridorState.get_indices_by_put_obstacle_number(index_of_available_act[i])
                if not QuoridorState.is_obstacle_puttable(QuoridorState.whose_turn, position_x, position_y, direction):
                    unavailable_act.append(index_of_available_act[i])


        abs_indices_of_available_acts = QuoridorState.get_relative_complement(index_of_available_act, unavailable_act)

        index_of_available_act = QuoridorState.convert_absolute_logits_index_by_relatives(abs_indices_of_available_acts, self.player_index)

        #print(index_of_available_act)

        # -------------------------------------
        # 차집합 계산후 선택 샘플
        available_logits = []
        #print(self.cashed_logits[0][index_of_available_act[0]])

        for i in range(len(index_of_available_act)):
            available_logits.append(logits[index_of_available_act[i]])

        #print(available_logits)
        hypothesis = tf.nn.softmax(available_logits)
        #print(hypothesis)

        # 행동선택함수가 합이 1인경우에만 쓸수 있도록 되어있어서
        # 오차가 생겨도 쓸수 있게 작성

        sum_of_prob = 0.0
        #prob = tf.random.uniform([1], 0, 1)
        prob = random.random()
        # print(prob)
        target_index = len(hypothesis) - 1

        for i in range(len(hypothesis)):
            if sum_of_prob < prob:
                target_index = i
            sum_of_prob += hypothesis[i]

        relative_prior_act = QuoridorState.convert_absolute_logit_index_by_relative(index_of_available_act[target_index], self.player_index)
        self.probability_for_prior_act = tf.cast(hypothesis[target_index], tf.float32)


        # -------------------------------------
        # 행동

        if relative_prior_act < 12:
            # 이동 인덱스로 바꾸는 함수
            position_x, position_y = QuoridorState.get_tile_index_by_move_player_index(relative_prior_act)
            QuoridorState.move_player(QuoridorState.whose_turn, position_x, position_y)

            if self.process_number == 0:
                print((self.player_index + 1).__str__() + f' player moved to '
                      + QuoridorState.players_coordinates[self.player_index].__str__())

        if relative_prior_act > 11:
            # 장애물 인덱스로 바꾸는 함수
            position_x, position_y, direction = QuoridorState.get_indices_by_put_obstacle_number(relative_prior_act)
            QuoridorState.put_obstacle(QuoridorState.whose_turn, position_x, position_y, direction)

            if self.process_number == 0:
                print((self.player_index + 1).__str__() + f' player put obstacle at ' + [position_x, position_y].__str__()
                      + direction.__str__())

        self.acted_before = True

        self.prior_act = index_of_available_act[target_index]
        #print(self.get_feature_vector_and_value_func.variables[-2][0])

        return


    def learn(self, theta_global, theta_v_global, g_pi_global, g_v_global):
        # 행동과 학습을 분리시켜 학습과정인 이 함수에서 CNN과의 연결 및 학습

        # ----------------------------
        # gradient tape를 써서 업데이트

        if not self.acted_before:
            return

        input_vector = self.get_input_vector()

        #next_feature_vector = next_CNN_outputs[0]
        #phi_next = np.transpose(next_feature_vector)

        #if self.process_number == 0:
        #    print(self.player_index + 1, self.cashed_input[0][381], input_vector[0][381], self.cashed_input[0][39],
        #          input_vector[0][39])

        if not QuoridorState.is_game_set():
            next_CNN_outputs = self.get_feature_vector_and_value_func(input_vector, training=False)
            value_func_next = next_CNN_outputs[1]
            #value_func_next = self.direct_value_func(input_vector, training=False)

        else:

            value_func_next = 0.0


        # delta를 어떻게 가공할지 고민필요
        # 일단은 심플하게 원 핫 인코딩
        #label = tf.nn.relu(delta)
        label = 1

        labels = np.array([0.0] * CNN.number_of_actions, dtype=float)
        labels[self.prior_act] = label

        with tf.GradientTape() as tape:

            CNN_outputs = self.get_feature_vector_and_value_func(self.cashed_input, training=True)
            #feature_vector = np.transpose(CNN_outputs[0][0])
            phi = np.transpose(CNN_outputs[0])
            #phi = np.transpose(self.cashed_input)[0]

            #logits = np.dot(np.transpose(self.theta_pi), feature_vector)

            value_func = CNN_outputs[1]
            #value_func = self.direct_value_func(self.cashed_input, training=True)

            delta = self.get_reward() + self.discount_factor * value_func_next - value_func
            v_loss = delta * delta
            #print(delta)
            #print(v_loss)

            with tf.GradientTape() as actor_tape:
                logits = self.get_logit(self.cashed_feature_vector, training=True)
                # print(tf.nn.softmax(logits))

                # 확률 합계 1 문제로 함수에 버그가 있는 모양이라 수제 함수로 대체
                # 수제 함수도 오버플로우 문제가 있는 모양
                actor_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits + 1e-10)
                # actor_loss = -tf.reduce_sum(labels * tf.math.log(tf.nn.softmax(logits) + 1e-10))

                if True in tf.math.is_nan(logits[0]):
                    print(self.get_logit.variables)

            loss = actor_loss + v_loss * 0.5

        if True in tf.math.is_nan(delta):
            print(self.discount_factor * value_func_next, value_func)


        #print(logits)
        #print(actor_loss)
        #print(self.get_logit.variables)
        # gradient로 뽑은 기울깃값은 주소값만을 복사해오는 것으로 보임
        # variables로 뽑은 리스트는 주솟값의 리스트로 보임

        grads = tape.gradient(loss, self.get_feature_vector_and_value_func.variables)
        #grads = tape.gradient(v_loss, self.direct_value_func.variables)

        actor_grads = actor_tape.gradient(actor_loss, self.get_logit.variables)
        #print(actor_grads[-2])
        #print(grads[-2])

        #print(np.dot(np.transpose(self.direct_value_func.variables[-1].numpy()), phi))

        #print(grads[-2][0][0], actor_grads[-2][0][0])
        if self.process_number == 0:
            print(delta, value_func, actor_loss, loss)

        #print(tf.nn.softmax(logits))

        # 마지막 변수는 바이어스이기 때문에
        updates_for_actor = actor_grads[-1] / self.probability_for_prior_act
        #updates_for_actor = actor_grads[-2] / tf.nn.softmax(logits[0])[self.prior_act]

        # ----------------------------

        self.dutch_trace_theta_pi = self.discount_factor * self.lambda_theta_pi * self.dutch_trace_theta_pi \
                                    + self.i_discount_factor * updates_for_actor
        #print(self.dutch_trace_theta_pi)
        self.dutch_trace_theta_v = self.discount_factor * self.lambda_theta_v * self.dutch_trace_theta_v + phi \
                                - self.alpha_theta_v * self.discount_factor * self.lambda_theta_v \
                                   * (np.dot(np.transpose(self.dutch_trace_theta_v), phi)) * phi
        #print(self.dutch_trace_theta_v)

        # theta_v의 업데이트 부분에 해당. 가비지 의심 필요
        grads[-1] = (delta + value_func - self.value_func_old) * self.dutch_trace_theta_v - (value_func - self.value_func_old) * phi * -1
        actor_grads[-1] = delta * self.dutch_trace_theta_pi

        #grads[-1] = grads[-1] * self.alpha_theta_v * 0.0001
        #print(grads[-1][381])

        # 계산한 theta_pi를 cnn 패러미터에 업데이트

        #value_func_opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        #value_func_opt.variables()[-1].assign(g_v_global[:])

        #print(self.get_feature_vector_and_value_func.variables[0][0][0][0][0])
        self.value_func_opt.apply_gradients(zip(grads, self.get_feature_vector_and_value_func.variables))
        #for i in range(len(self.get_feature_vector_and_value_func.variables)):
        #    self.get_feature_vector_and_value_func.variables[i].assign(self.get_feature_vector_and_value_func.variables[i].numpy() - grads[i] * 0.01)
        #print(self.get_feature_vector_and_value_func.variables[0][0][0][0][0])

        #value_func_opt.apply_gradients(zip(grads, self.direct_value_func.variables))

        #print(value_func_opt.variables()[-1])

        #opt = tf.keras.optimizers.RMSprop(learning_rate=self.alpha_theta_v)
        #opt.apply_gradients(zip(grads, self.get_feature_vector_and_value_func.variables))

        #print(grads[-1])
        #print(self.direct_value_func.variables[-2])
        #print(self.direct_value_func.variables[-1][381])
        #print(grads[-1][381], 0.1 * grads[-1][381] * grads[-1][381], value_func_opt.variables()[-1][381])
        #print(self.direct_value_func.variables[-1][381])

        #self.direct_value_func.variables[-1].assign(self.direct_value_func.variables[-1].numpy() + grads[-1] * self.alpha_theta_v)

        if True in tf.math.is_nan(logits[0]):
            print(tf.nn.softmax(logits[0])[self.prior_act])

        #actor_opt.apply_gradients(zip(actor_grads, self.get_logit.variables))

        self.get_logit.variables[-1].assign(self.get_logit.variables[-1].numpy() - actor_grads[-1] * self.alpha_theta_pi)

        #print(self.get_logit.variables[-2][0][0])
        #print(self.get_feature_vector_and_value_func.variables[-2][0])


        self.i_discount_factor = self.discount_factor * self.i_discount_factor
        self.value_func_old = value_func_next
        #phi = phi_next
        self.cashed_input = input_vector

        #print(theta_v_global[381])

        self.update_theta_global_using_RMSProp(theta_global, theta_v_global, g_pi_global, g_v_global, grads[-1])

        self.thread_step_counter += 1


        #print(theta_v_global[381])

        return


    def get_input_vector(self):
        # atari는 210 * 160
        #input_vector = QuoridorState.get_obstacles_in_board_by_binary(self.player_index)
        #input_vector = QuoridorState.get_additional_feature_vector(input_vector, self.player_index)

        #if self.process_number == 0:
        #    for i in range(60):
        #        print(expanded_input_vector[i])

        #if self.process_number == 0:
        #    for i in range(CNN.number_of_input_side):
        #        print(input_vector[i])

        input_vector = self.get_atari_size_input_vector()

        returned_input = tf.reshape(input_vector, [-1, CNN.number_of_input_side, CNN.number_of_input_side, 1])
        return returned_input

    def get_reward(self):
        # 누구의 턴인지 구분해 보상
        if QuoridorState.is_game_set():
            if QuoridorState.winner_index == self.player_index:
                return 1.0
            else:
                return -1.0
        return 0.0


    def update_theta_global_using_RMSProp(self, theta_global, theta_v_global, g_pi_global, g_v_global, value_grads):
        #print(self.get_feature_vector_and_value_func.variables[-2])
        #
        # model.variables에서 텐서를 직접 끌어 쓰면 속도가 굉장히 느리며 가비지가 빨리쌓임
        # 넘파이 배열로 변경해줘야 해결됨. 이유는 모르겠음
        #theta_v = self.get_feature_vector_and_value_func.variables[-2].numpy()
        #theta_v = self.direct_value_func.variables[-1].numpy()

        #print(theta_v_global[381])
        #for i in range(len(g_v_global)):
            #g_v_global[i] = self.moving_average_parameter * g_v_global[i] + (1 - self.moving_average_parameter) * (theta_v[i] ** 2)
            #theta_v_global[i] = theta_v_global[i] - self.initial_learning_rate * theta_v[i] / np.sqrt(g_v_global[i] + self.epsilon)
            #g_v_global[i] = self.moving_average_parameter * g_v_global[i] + (1 - self.moving_average_parameter) * (value_grads[-1][i] ** 2)
            #theta_v_global[i] = theta_v_global[i] - self.initial_learning_rate * value_grads[-1][i] / np.sqrt(g_v_global[i] + self.epsilon)
            #theta_v_global[i] = theta_v_global[i] + value_grads[-1][i] / 2
            #theta_v_global[i] = theta_v[i]



        theta_v_global[:] = theta_v_global[:] + np.transpose(value_grads)[0] * 0.5 * self.alpha_theta_v
        #print(theta_v_global[381])

        #print(theta_v_global[0])
        self.get_feature_vector_and_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))
        #self.direct_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))

        #print(theta_v_global[0])

        if self.thread_step_counter % self.t_max == 0 or QuoridorState.is_game_set():
            if self.process_number == 0:
                print(f'RMSProp')

            theta_pi = self.get_logit.variables[-1].numpy()
            # print(theta_pi)

            flattened_theta_pi = np.array([0.0] * CNN.number_of_feature_vector * CNN.number_of_actions)
            for i in range(CNN.number_of_feature_vector):
                for ii in range(CNN.number_of_actions):
                    flattened_theta_pi[i * CNN.number_of_actions + ii] = theta_pi[i][ii]

            delta_theta_pi = theta_global[:] - flattened_theta_pi
            #print(theta_global[0])

            g_pi_global[:] = self.moving_average_parameter * np.array(g_pi_global[:]) \
                             + (1 - self.moving_average_parameter) * delta_theta_pi * delta_theta_pi
            theta_global[:] = theta_global[:] - self.initial_learning_rate * delta_theta_pi / np.sqrt(np.array(g_pi_global[:]) + self.epsilon)

            """
            for i in range(CNN.number_of_feature_vector):
                for ii in range(CNN.number_of_actions):
                    flattened_index = i * CNN.number_of_actions + ii
                    g_pi_global[flattened_index] = self.moving_average_parameter * g_pi_global[flattened_index] \
                                                   + (1 - self.moving_average_parameter) * (theta_pi[i][ii] ** 2)
                    theta_global[flattened_index] = theta_global[flattened_index] - self.initial_learning_rate * \
                                                    theta_pi[i][ii] \
                                                    / np.sqrt(g_pi_global[flattened_index] + self.epsilon)
            """

            #print(theta_global[0])

            self.Async_theta_pi_by_theta_pi_global(theta_global, theta_v_global, g_pi_global, g_v_global)

            self.thread_step_counter = 1

        return


    def Async_theta_pi_by_theta_pi_global(self, theta_global, theta_v_global, g_pi_global, g_v_global):
        theta_pi_global = theta_global[:]
        theta_pi = np.array([[0.0 for col in range(CNN.number_of_actions)] for row in range(CNN.number_of_feature_vector)])

        for i in range(CNN.number_of_feature_vector):
            for ii in range(CNN.number_of_actions):
                theta_pi[i][ii] = theta_pi_global[i * CNN.number_of_actions + ii]


        #self.theta_pi = theta_pi
        #print(self.get_logit.variables)
        self.get_logit.variables[-1].assign(theta_pi)

        return


    def get_atari_size_input_vector(self):

        expand_scale = 5
        expanded_input_vector = [[0.0 for col in range(CNN.number_of_input_side)]
                                                    for row in range(CNN.number_of_input_side)]

        input_vector = QuoridorState.get_obstacles_in_board_by_binary(self.player_index)

        for i in range(17):
            for ii in range(17):
                if input_vector[i][ii] == 1.0:                    
                    for iii in range(expand_scale):
                        for iiii in range(expand_scale):
                            expanded_input_vector[i * expand_scale + iii][ii * expand_scale + iiii] = 1.0
                            
        players_obstacles_number = QuoridorState.players_obstacles[self.player_index]
        enemys_obstacles_number = QuoridorState.players_obstacles[1 - self.player_index]
        
        for i in range(5):            
            for ii in range(players_obstacles_number):
                expanded_input_vector[95 + i][3 * ii] = 1.0
            for ii in range(enemys_obstacles_number):
                expanded_input_vector[3 * ii][95 + i] = 1.0


        players_coord = QuoridorState.players_coordinates[self.player_index]
        enemy_coord = QuoridorState.players_coordinates[1 - self.player_index]

        if self.player_index == 0:
            for i in range(5):
                expanded_input_vector[players_coord[0] * expand_scale + 2][players_coord[1] * expand_scale + i] = 1.0
                expanded_input_vector[players_coord[0] * expand_scale + i][players_coord[1] * expand_scale] = 1.0
                expanded_input_vector[enemy_coord[0] * expand_scale + 2][enemy_coord[1] * expand_scale + i] = 1.0
                expanded_input_vector[enemy_coord[0] * expand_scale + i][enemy_coord[1] * expand_scale + 4] = 1.0

        else:
            players_coord = [16 - players_coord[0], 16 - players_coord[1]]
            enemy_coord = [16 - enemy_coord[0], 16 - enemy_coord[1]]
            for i in range(5):
                expanded_input_vector[enemy_coord[0] * expand_scale + 2][enemy_coord[1] * expand_scale + i] = 1.0
                expanded_input_vector[enemy_coord[0] * expand_scale + i][enemy_coord[1] * expand_scale + 4] = 1.0
                expanded_input_vector[players_coord[0] * expand_scale + 2][players_coord[1] * expand_scale + i] = 1.0
                expanded_input_vector[players_coord[0] * expand_scale + i][players_coord[1] * expand_scale] = 1.0

        #for i in range(100):
        #    print(expanded_input_vector[i])

        return expanded_input_vector


