import csv
import math as m
import matplotlib.pyplot as plt
dataFileName = '../data/v0_2_midnight_copy.csv'



class Drawer():
    save_path = './pathes_check/'

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
    cnt = 0
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

        cnt += 1
        #if not cnt%1:
        if episode_num is 1:
            Drawer.drawEpisodePath(episode_ls[-1]['episode'][-1], episode_ls[-1]['sec'][-1], episode_ls[-1]['path'][-1])
    print('n episode : {}'.format(len(episode_ls)))

    state_size = 3 + 4 + 8
    action_size = 2
    for episode in episode_ls:
        total_step = episode['sec'][-1] + 1

        for i in range(total_step):
            idx_t2 = i - 10
            idx_t1 = i - 5
            idx_t = i
            idx_t_next = i + 1
            if idx_t2 < 0:
                continue
            if idx_t_next >= total_step:
                break

            state = (episode['cur_vel'][idx_t2], episode['cur_vel'][idx_t1], episode['cur_vel'][idx_t], \
                     episode['cur_tire_angle'][idx_t2], episode['cur_vel'][idx_t1], episode['cur_vel'][idx_t], \
                    episode['path'][idx_t][0][0], episode['path'][idx_t][0][1], episode['path'][idx_t][0][2], episode['path'][idx_t][0][4], \
                    episode['path'][idx_t][1][0], episode['path'][idx_t][1][1], episode['path'][idx_t][1][2], episode['path'][idx_t][1][4])
            action = (episode['cmd_vel'][idx_t], ['cmd_w'][idx_t])

            #reward calculation
            v_gap = (episode['cur_vel'][idx_t_next] - episode['path'][idx_t_next][0][0]) ** 2
            #)