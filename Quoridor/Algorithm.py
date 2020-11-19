# 알고리즘 내의 전역변수를 관리
# A3C의 비동기 프로세스를 Agent 스크립트를 통해 제어
# QuoridorState의 규칙에 따라 게임을 학습하는 에이전트의 순서를 관리
# 학습된 패러미터의 출력 및 저장도 이곳에서 관리

from threading import Thread
import asyncio
from multiprocessing import Process, current_process
import threading

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import multiprocessing
from multiprocessing import Process, Value, Array

import Agent
import CNN
import QuoridorState

import test


alpha_theta_v = 0.001

# 전역변수

number_of_players = 2

# 난수화 및 이동패러미터 강화 고민 필요
theta_global = Array('f', [0.1] * CNN.number_of_feature_vector * CNN.number_of_actions, lock=False)
theta_v_global = Array('f', [0.0] * CNN.number_of_feature_vector, lock=False)
#theta_global = np.ctypeslib.as_array(Array('f', [0.1] * CNN.number_of_feature_vector * CNN.number_of_actions))
#theta_v_global = np.ctypeslib.as_array(Array('f', [0.0] * CNN.number_of_feature_vector, lock=False))

# RMSProp에서 쓰일 전역 공유 변수 moving average of elementwise squared gradients
g_pi_global = Array('f', [0.0] * CNN.number_of_feature_vector * CNN.number_of_actions, lock=False)
g_v_global = Array('f', [0.0] * CNN.number_of_feature_vector, lock=False)

value_func_opt = tf.keras.optimizers.RMSprop(learning_rate=alpha_theta_v)
feature_vector_and_value_func = CNN.create_feature_vector_and_value_func_model()
# -------------------------------------

test_sum = Value('i', 0)

fully_learned = False
number_of_game = 100000
number_of_games_to_be_saved = 1

is_set_to_load_parameters = False
is_set_to_save_parameters = True


# 학습이 시작되어 멀티프로세스를 초기화
def init_multi_process():

    global value_func_opt
    #global feature_vector_and_value_func

    load_or_init_global_parameter()

    # 최적의 cpu갯수 찾을 필요 있음
    c = 4

    procs = []
    for i in range(c):
        p = multiprocessing.Process(target=init_process, args=(theta_global, theta_v_global, g_pi_global, g_v_global, i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def load_or_init_global_parameter():

    global is_set_to_load_parameters
    global theta_global
    global theta_v_global
    global g_pi_global
    global g_v_global
    global value_func_opt
    global feature_vector_and_value_func

    if is_set_to_load_parameters:

        theta_global = Array('f', np.loadtxt("theta_global.txt"), lock=False)
        theta_v_global = Array('f', np.loadtxt("theta_v_global.txt"), lock=False)

        g_pi_global = Array('f', np.loadtxt("g_pi_global.txt"), lock=False)
        g_v_global = Array('f', np.loadtxt("g_v_global.txt"), lock=False)

        value_func_opt = tf.keras.optimizers.RMSprop(learning_rate=alpha_theta_v)

        feature_vector_and_value_func = tf.keras.models.load_model(
            "get_feature_vector_and_value_func0.h5", compile=True)
    else:
        return


def init_process(theta_global, theta_v_global, g_pi_global, g_v_global, process_number):
    # 이 함수 이하 개별적인 프로세스. 따라서 전역스크립트를 사용해도 프로세스끼리 공유되지는 않음.
    # 전역변수는 전용 멀티프로세스 라이브러리를 통해서 따로 관리.

    global fully_learned
    global feature_vector_and_value_func

    agents = []

    for i in range(number_of_players):
        agents.append(Agent.Agent(theta_global, theta_v_global, g_pi_global, g_v_global, process_number))


    # 임시로 모델 로드를 여기에
    if is_set_to_load_parameters:
        
        feature_vector_and_value_func = tf.keras.models.load_model(
            "get_feature_vector_and_value_func0.h5", compile=True)

    agents[0].get_feature_vector_and_value_func = feature_vector_and_value_func
    agents[1].get_feature_vector_and_value_func = feature_vector_and_value_func

    agents[0].get_feature_vector_and_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))
    agents[1].get_feature_vector_and_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))

    #agents[0].direct_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))
    #agents[1].direct_value_func.variables[-1].assign(np.transpose([theta_v_global[:]]))

    #value_func_opt = tf.keras.optimizers.RMSprop(learning_rate=agents[0].alpha_theta_v)

    #agents[0].value_func_opt = value_func_opt
    #agents[1].value_func_opt = value_func_opt

    episode_count = 0

    while not fully_learned:
        QuoridorState.init_game(number_of_players)
        loop_over_episodes(agents, theta_global, theta_v_global, g_pi_global, g_v_global)

        episode_count += 1
        print(episode_count.__str__() + f' game passed')

        if agents[0].process_number == 0:
            if episode_count % number_of_games_to_be_saved == 0:
                # 전역 변수 저장
                save_parameter(theta_global, theta_v_global, g_pi_global, g_v_global)

                # 임시 모델 저장
                #if is_set_to_save_parameters:

                    #agents[0].get_feature_vector_and_value_func.save("get_feature_vector_and_value_func0.h5")
                    #agents[0].direct_value_func.save("direct_value_func0.h5")

        if episode_count > number_of_game:
            fully_learned = True


def loop_over_episodes(agents, theta_global, theta_v_global, g_pi_global, g_v_global):

    for i in range(number_of_players):
        agents[i].init_for_episode(i)

    move_count = 0
    #QuoridorState.change_turn()

    while not QuoridorState.is_game_set():
        # 학습과정에서 2개이상의 에이전트가 대전하는 경우
        # 에이전트가 가지는 패러미터가 다르므로
        # 하나의 특징벡터는 두 에이전트에 공유되지 않는다.
        # with tf.GradientTape() as tape의 그레디언트를 인덴트 밖에서 계산할
        # 방법이 없어 추가 메모리와 연산시간을 사용하는 쪽으로 구현.

        # 행동
        # input벡터를 받아 소프트맥스를 계산하고 행동을 취하는 과정은 Agent클래스로.
        agents[QuoridorState.whose_turn].act_by_policy()

        QuoridorState.change_turn()

        # 학습
        # 먼저 행동에의해 게임이 종료되었는지 체크
        # 행동 후의 특징벡터로 다음 턴 플레이어의 이전 행동에 대한 패러미터 벡터를 업데이트

        if QuoridorState.is_game_set():
            #for agent in agents:
            #    agent.learn(theta_global, theta_v_global, g_pi_global, g_v_global)
            #    QuoridorState.change_turn()

            agents[QuoridorState.whose_turn].learn(theta_global, theta_v_global, g_pi_global, g_v_global)
            QuoridorState.change_turn()
            agents[QuoridorState.whose_turn].learn(theta_global, theta_v_global, g_pi_global, g_v_global)

            if agents[QuoridorState.whose_turn].process_number == 0:
                print(f'player' + (QuoridorState.whose_turn + 1).__str__() + f' won by ' + move_count.__str__() + f' moves')
            return

        #if agents[QuoridorState.whose_turn].process_number == 0:
        #    QuoridorState.print_abs_map_status()

        # 후처리
        agents[QuoridorState.whose_turn].learn(theta_global, theta_v_global, g_pi_global, g_v_global)

        next_index = QuoridorState.get_next_players_index(QuoridorState.whose_turn)
        #agents[next_index].get_logit.variables[-1].assign(agents[QuoridorState.whose_turn].get_logit.variables[-1].numpy())

        #if agents[QuoridorState.whose_turn].process_number == 0:
        #    print(QuoridorState.get_obstacles_in_board_by_binary()[19][19])

        move_count = move_count + 1
        # 임시로 행동 후 턴 표기 변경을 여기서


        #임시로
        #return


def save_parameter(theta_global, theta_v_global, g_pi_global, g_v_global):

    #a = [0, 1, 2]
    #print(g_v_global[0])
    global feature_vector_and_value_func

    if is_set_to_save_parameters:
        np.savetxt("theta_global.txt", theta_global[:])
        np.savetxt("theta_v_global.txt", theta_v_global[:])
        np.savetxt("g_pi_global.txt", g_pi_global[:])
        np.savetxt("g_v_global.txt", g_v_global)

        feature_vector_and_value_func.save("get_feature_vector_and_value_func0.h5")

    return

