import csv
import math as m
import matplotlib.pyplot as plt
import numpy as np
from array import array


dataFileName = './v0_2_midnight_copy.csv'
binaryFileName = './binary_data.tw'

#state calculation members
MINIMUM_LOOKAHEAD_DISTANCE = 1
N_GOAL_POSE = 3

#reward calculation members
COEF_V = 2
COEF_TH = 6 #these params make errors similar distribution
W_V = 1
W_TH = 3

#바이너리로 저장하기
# 불러와서 학습시키기
# v, w 적합성 판단하기 -> 교수님께 한번 검증받기
# 실제 차에 넣고 학습돌리기
# 개선할만한 아이디어(physics)

state_size = 3 + 3 + 1 + 2 * 3  # velocity, theta_tire, target_vel, goal xy * 3
action_size = 2


class BinaryRecorder():
    def __init__(self):
        # save data in binary mode. 5 float(state, action, reward, next_state, done)
        self.record = open(binaryFileName, 'wb')
    def save(self, state, action, reward, next_state, done):
        ary = array('f', state)
        ary.tofile(self.record)
        ary = array('f', action)
        ary.tofile(self.record)
        ary = array('f', [reward])
        ary.tofile(self.record)
        ary = array('f', next_state)
        ary.tofile(self.record)
        ary = array('f', [done])
        ary.tofile(self.record)


class Drawer():
    save_path = './pathes_check/'
    state_goals_path = './states_goals_check/'
    def drawEpisodePathAndStateGoals(episode, sec, path, goals):
        plt.clf()
        x = []
        y = []
        for p in path:
            x.append(-p[2])
            y.append(p[1])  # rotate 90 degree

        plt.scatter(x, y)
        for goal in goals:
            plt.plot(-goal[1], goal[0], 'ro')

        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

        plt.title(str(episode) + '_' + str(sec))

        axes = plt.gca()
        axes.set_xlim([-2.0, 2.0])
        axes.set_ylim([-0.5, 10.0])
        #plt.show()
        plt.savefig(Drawer.state_goals_path + str(episode) + '_' + str(sec) + '.png')


    def drawEpisodePath(episode, sec, path):
        plt.clf()
        x = []
        y = []
        vec_data = []
        for p in path:
            x.append(-p[2])
            y.append(p[1]) #rotate 90 degree

            th = p[4] + m.pi/2
            mag = p[0]*0.1
            vec_data.append((-p[2], -p[2] + mag*m.cos(th), p[1], p[1] + mag*m.sin(th)))

        plt.scatter(x, y)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.title(episode)
        for vec in vec_data:
            plt.plot((vec[0], vec[1]), (vec[2], vec[3]), 'b')
        axes = plt.gca()
        axes.set_xlim([-1.0, 1.0])
        plt.savefig(Drawer.save_path + str(episode) + '_' + str(sec) + '.png')

if __name__=='__main__':
    f = open(dataFileName, 'r')
    rdr = csv.reader(f)

    episode_ls = []
    prev_episode = ''
    CSV_FIELD_NAMES = ['sec','stamp','cur_vel','cur_tire_angle','cmd_vel','cmd_w','cur_x','cur_y','cur_z','x_cur_ang_vel','y_cur_ang_vel','z_cur_ang_vel' \
        ,'x_cur_angle','y_cur_angle','z_cur_angle','x_cur_acc','y_cur_acc','z_cur_acc','path','episode']
    episode_num = 0
    for line in rdr:
        #check episode number
        cur_episode = line[0]
        if cur_episode != prev_episode:
            episode_num += 1

            episode_ls.append({})
            for field in CSV_FIELD_NAMES:
                episode_ls[-1][field] = []
        prev_episode = cur_episode
        episode_ls[-1]['episode'].append(episode_num)
        episode_ls[-1]['sec'].append(int(line[1]))
        episode_ls[-1]['stamp'].append(float(line[2]))
        episode_ls[-1]['cur_vel'].append(float(line[3]))
        episode_ls[-1]['cur_tire_angle'].append(float(line[4]))
        episode_ls[-1]['cmd_vel'].append(float(line[5]))
        episode_ls[-1]['cmd_w'].append(float(line[6]))
        episode_ls[-1]['cur_x'].append(float(line[7]))
        episode_ls[-1]['cur_y'].append(float(line[8]))
        episode_ls[-1]['cur_z'].append(float(line[9]))
        episode_ls[-1]['x_cur_ang_vel'].append(float(line[10]))
        episode_ls[-1]['y_cur_ang_vel'].append(float(line[11]))
        episode_ls[-1]['z_cur_ang_vel'].append(float(line[12]))
        episode_ls[-1]['x_cur_angle'].append(float(line[13]))
        episode_ls[-1]['y_cur_angle'].append(float(line[14]))
        episode_ls[-1]['z_cur_angle'].append(float(line[15]))
        episode_ls[-1]['x_cur_acc'].append(float(line[16]))
        episode_ls[-1]['y_cur_acc'].append(float(line[17]))
        episode_ls[-1]['z_cur_acc'].append(float(line[18]))

        episode_ls[-1]['path'].append([])
        for i in range(19, len(line), 5):
            d = (float(line[i]), float(line[i+1]),float(line[i+2]),float(line[i+3]), float(line[i+4])) #v, x, y, yaw
            episode_ls[-1]['path'][-1].append(d)

        # Drawer.drawEpisodePath(episode_ls[-1]['episode'][-1], episode_ls[-1]['sec'][-1], episode_ls[-1]['path'][-1])
    print('n episode : {}'.format(len(episode_ls)))

    file_v_and_th = open('./v_and_th.csv', 'w')
    recorder = BinaryRecorder()

    for episode in episode_ls:
        total_step = episode['sec'][-1] + 1
        state = None
        for i in range(total_step):
            idx_t2 = i - 10 + 1
            idx_t1 = i - 5 + 1
            idx_t = i + 1
            idx_t_next = i + 2
            if idx_t2 < 0 or idx_t % 2:
                continue
            if idx_t_next >= total_step:
                break

            ''' calculate 3 goal point proportional to velocity '''
            goals = []
            cur_vel = episode['cur_vel'][idx_t]

            target_dists = []
            d = cur_vel * 1 # d = vt : 1.0d, 1.5d, 2.0d, ...
            if d < MINIMUM_LOOKAHEAD_DISTANCE:
                target_dists = [MINIMUM_LOOKAHEAD_DISTANCE * (1+0.5*i) for i in range(N_GOAL_POSE)]
            else:
                target_dists = [d * (1+0.5*i) for i in range(N_GOAL_POSE)]

            for target_dist in target_dists:
                #find 2 points. one has lower dist than target dist, and the other case for other point.
                p1 , p2 = -1, -1
                for j in range(1, len(episode['path'][idx_t])):
                    goal_cur = episode['path'][idx_t][j]
                    dist_cur = m.sqrt(goal_cur[1]**2 + goal_cur[2]**2)

                    if dist_cur > target_dist:
                        p1 = (episode['path'][idx_t][j-1][1], episode['path'][idx_t][j-1][2])
                        p2 = (goal_cur[1], goal_cur[2])
                        break

                if p1==-1 or p2==-1:
                    raise Exception('can not find p1 or p2. pathes list is broken')

                # calculate the line equation for 2 points
                is_vertical = False
                slope = 0
                if m.fabs(p2[0] - p1[0]) < 0.00000001:
                    is_vertical = True
                    slope = p2[1]
                else:
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    b = p2[1] - slope*p2[0]

                #find the goal whose dist is target_dist
                r1, r2 = ((-2*slope*b) + m.sqrt(4*slope**2*b**2 - 4*(slope**2+1)*(b**2-target_dist**2))) / (2*(slope**2+1)), \
                    ((-2*slope*b) + m.sqrt(4*slope**2*b**2-4*(slope**2+1)*(b**2-target_dist**2))) / (2*(slope**2+1))
                y1, y2 = slope*r1+b, slope*r2+b
                dist_to_prev = m.sqrt((r1-p1[0])**2 + (y1-p1[1])**2)
                if dist_to_prev < target_dist:
                    goals.append((r1, y1))
                else:
                    goals.append((r2, y2))
            ''' calculate 3 goal point proportional to velocity end '''
            #if episode['episode'][idx_t] <= 2:
            #    Drawer.drawEpisodePathAndStateGoals(episode['episode'][idx_t], episode['sec'][idx_t], \
            #                                    episode['path'][idx_t], goals)

            target_vel = episode['path'][idx_t][0][0]
            next_state = (episode['cur_vel'][idx_t2], episode['cur_vel'][idx_t1], episode['cur_vel'][idx_t], \
                     episode['cur_tire_angle'][idx_t2], episode['cur_tire_angle'][idx_t1], episode['cur_tire_angle'][idx_t], \
                     episode['path'][idx_t][0][0], \
                    goals[0][0], goals[0][1], goals[1][0], goals[1][1], goals[2][0], goals[2][1])
            assert len(next_state) == state_size


            action = (episode['cmd_vel'][idx_t], episode['cmd_w'][idx_t])

            ''' reward calculation '''
            v_square = (target_vel - episode['cur_vel'][idx_t]) ** 2
            # calc direction error with ordinary least square method
            A = np.array(goals)
            A[:,1] = 1
            B = np.zeros(len(goals)).reshape(-1,1)
            for i in range(len(goals)):
                B[i] = goals[i][1]
            lsm_line = np.linalg.inv((np.matmul(np.transpose(A), A)))
            lsm_line = np.matmul(lsm_line, np.transpose(A))
            lsm_line = np.matmul(lsm_line, B)
            lsm_slope = lsm_line[0][0]
            theta_positions = m.atan(lsm_slope)
            th_square = theta_positions**2

            reward = W_V * m.exp(-COEF_V * v_square) + W_TH * m.exp(-COEF_TH * th_square)
            file_v_and_th.write(str(episode['sec'][idx_t]) + ',' + str(v_square) + \
                                ',' + str(th_square) + ',' + str(reward) + '\n')
            if state == None:
                state = next_state
                continue

            recorder.save(state, action, reward, next_state, False)

            state = next_state
