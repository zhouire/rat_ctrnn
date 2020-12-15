import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def generate_path(n, k, max_delta_rot, speed_range, x_init, y_init, xy_range):
    # initialize rotation
    rot_init = np.random.random() * 2 * np.pi
    rot = [rot_init]

    # initialize speed
    #delta_speed = [delta_speed_init]
    speed = [np.mean(speed_range)]

    if x_init is not None and y_init is not None:
        # generate positions
        x = [x_init]
        y = [y_init]
    else:
        x = [np.random.random()-0.5]
        y = [np.random.random()-0.5]

    for i in range(n):
        ### DETERMINING SPEED
        '''
        speed_valid = False
        # keep generating until we get a valid new speed
        while not speed_valid:
            #print("speed invalid")
            #new_delta_speed = np.random.random() * max_delta_speed
            #new_speed = speed[-1] + new_delta_speed
            new_speed = np.random.random() * speed_range[1]
    
            if speed_range[0] < new_speed < speed_range[1]:
                speed_valid = True
        '''
        new_speed = np.random.random() * (speed_range[1] - speed_range[0]) + speed_range[0]

        # interpolate speed
        new_speed_interp = np.interp(np.arange(k+1), [0, k], [speed[-1], new_speed])[1:]
        speed += new_speed_interp.tolist()


        ### DETERMINING ROTATION
        # get new rotation
        new_delta_rot = np.random.randn() * max_delta_rot
        new_rot = rot[-1] + new_delta_rot

        # interpolate rotation
        new_rot_interp = np.interp(np.arange(k+1), [0, k], [rot[-1], new_rot])[1:]

        '''
        if np.random.random() > 0.5:
            new_rot_interp = np.interp(np.arange(k), [0, k], [rot[-1], new_rot])
        else:
            new_rot_interp = np.interp(np.arange(k), [0, k], [rot[-1], new_rot-2*np.pi])
        '''

        # calculate x and y, test to see if any part of the path is outside of [-1, 1]
        new_delta_x = new_speed_interp * np.cos(new_rot_interp)
        new_delta_y = new_speed_interp * np.sin(new_rot_interp)

        new_x = x[-1] + np.cumsum(new_delta_x)
        new_y = y[-1] + np.cumsum(new_delta_y)
        #new_x = new_x[1:]
        #new_y = new_y[1:]

        # check to see if all new_x and new_y values are in range [-1,1]
        rot_invalid = sum(new_x <= -1) > 0 or sum(new_x >= 1) > 0 or sum(new_y <= -1) > 0 or sum(new_y >= 1) > 0

        if rot_invalid:
            # resample angle until everything's valid
            # we get stuck if we go through the loop at least once, and problem_idx is 0
            loopcount = 0

            while rot_invalid:
                #print("rotation invalid")
                # if we hit a wall, make the angle parallel to the wall where we hit it and resample angle for the rest of the sample
                # keep the unproblematic parts
                problem_idx = len(new_x)
                for j in range(len(new_x)):
                    if not (xy_range[0] < new_x[j] < xy_range[1] and xy_range[0] < new_y[j] < xy_range[1]):
                        problem_idx = j
                        problem_x = new_x[j]
                        problem_y = new_y[j]
                        problem_angle = new_rot_interp[j]
                        break

                stuck = False
                if loopcount > 0 and problem_idx == 0:
                    stuck = True

                #print(problem_idx, len(new_x), len(new_y))
                if problem_idx > 0:
                    prev_rot = new_rot_interp[problem_idx-1]
                else:
                    prev_rot = rot[-1]
                    #print(problem_x, problem_y)
                    #print("prev_rot: ", prev_rot)

                x += new_x[:problem_idx].tolist()
                y += new_y[:problem_idx].tolist()
                rot += new_rot_interp[:problem_idx].tolist()
                new_speed_interp = new_speed_interp[problem_idx:]

                #new_rot_interp = new_rot_interp[problem_idx:]

                #new_x = []
                #new_y = []

                # take the angle parallel to the wall with the smallest change from the current angle
                if new_y[problem_idx] <= -1:
                    cond = prev_rot % (2*np.pi) > 3*np.pi/2 if not stuck else prev_rot % (2*np.pi) <= 3*np.pi/2
                    if cond:
                        new_angle = (prev_rot // (2*np.pi) + 1) * 2*np.pi
                        future_dir = 1
                    else:
                        new_angle = (prev_rot // (2*np.pi)) * 2*np.pi + np.pi
                        future_dir = -1

                elif new_y[problem_idx] >= 1:
                    cond = prev_rot % (2*np.pi) < np.pi/2 if not stuck else prev_rot % (2*np.pi) >= np.pi/2
                    if cond:
                        new_angle = (prev_rot // (2*np.pi)) * 2*np.pi
                        future_dir = -1
                    else:
                        new_angle = (prev_rot // (2*np.pi)) * 2*np.pi + np.pi
                        future_dir = 1

                elif new_x[problem_idx] <= -1:
                    cond = prev_rot % (2 * np.pi) < np.pi if not stuck else prev_rot % (2 * np.pi) >= np.pi
                    if cond:
                        new_angle = (prev_rot // (2 * np.pi)) * 2*np.pi + np.pi/2
                        future_dir = -1
                    else:
                        new_angle = (prev_rot // (2 * np.pi)) * 2 * np.pi + 3*np.pi/2
                        future_dir = 1

                elif new_x[problem_idx] >= 1:
                    cond = prev_rot % (2 * np.pi) < np.pi if not stuck else prev_rot % (2 * np.pi) >= np.pi
                    if cond:
                        new_angle = (prev_rot // (2 * np.pi)) * 2*np.pi + np.pi/2
                        future_dir = 1
                    else:
                        new_angle = (prev_rot // (2 * np.pi)) * 2*np.pi + 3*np.pi/2
                        future_dir = -1

                else:
                    raise ValueError("wtf just happened")

                future_delta_rot = future_dir * np.abs(np.random.randn() * max_delta_rot)
                interp_remain = len(new_x) - problem_idx
                new_rot_interp = np.interp(np.arange(interp_remain), [0, interp_remain-1], [new_angle, new_angle+future_delta_rot])
                #print("new rot interp: ", new_angle/np.pi, future_delta_rot)
                #print()

                # calculate x and y, test to see if any part of the path is outside of [-1, 1]
                new_delta_x = new_speed_interp * np.cos(new_rot_interp)
                new_delta_y = new_speed_interp * np.sin(new_rot_interp)

                new_x = x[-1] + np.cumsum(new_delta_x)
                new_y = y[-1] + np.cumsum(new_delta_y)
                # new_x = new_x[1:]
                # new_y = new_y[1:]
                #print(new_x, new_y)

                # check to see if all new_x and new_y values are in range [-1,1]
                rot_invalid = sum(new_x <= -1) > 0 or sum(new_x >= 1) > 0 or sum(new_y <= -1) > 0 or sum(new_y >= 1) > 0
                #print("rotation still invalid")

                loopcount += 1

            # after leaving the while loop, append the final new_x and new_y and rotation values
            x += new_x.tolist()
            y += new_y.tolist()
            rot += new_rot_interp.tolist()

        # if rotation is valid, append new position values, and new rotation values
        else:
            #print(len(x))
            # if this rotation works, append the new x and y values
            x += new_x.tolist()
            y += new_y.tolist()
            rot += new_rot_interp.tolist()

    return x, y, speed, rot

# generate a bunch of paths, concatenated end to end
def generate_movement(num_paths, n, k, max_delta_rot, speed_range, x_init, y_init, xy_range, save_path=None):
    x_all = []
    y_all = []
    speed_all = []
    rot_all = []

    for i in range(num_paths):
        x, y, speed, rot = generate_path(n, k, max_delta_rot, speed_range, x_init, y_init, xy_range)
        x_all += x[:-1]
        y_all += y[:-1]
        speed_all += speed[:-1]
        rot_all += rot[:-1]

    # scale rot_all to [-1, 1]
    rot_all = (np.array(rot_all) + np.pi) % (2*np.pi) - np.pi

    # scale speed to [0, 1]
    speed_all = np.array(speed_all)/speed_range[1]

    # create input_x and input_y; input_x is speed and rotation, input_y is position
    input_x = np.array([speed_all, rot_all]).T
    input_y = np.array([x_all, y_all]).T

    # save path data
    if save_path is not None:
        save_data = {"x": input_x, "y": input_y}
        pickle.dump(save_data, open(save_path, "wb"))

    return input_x, input_y


n = 25
k = 20
# angle is in radians
# this is not actually the max rotation, it's actually standard deviation for Gaussian generation
max_delta_rot = 0.8
#max_delta_speed = 0.001
speed_range = [0, 0.02]

#delta_speed_init = 0
x_init = 0
y_init = 0

xy_range = [-1, 1]
num_paths = int(1000000/500)
save_path = "rat_ctrnn_data_randstart.p"

x, y, speed, rot = generate_path(n, k, max_delta_rot, speed_range, x_init, y_init, xy_range)
#input_x, input_y = generate_movement(num_paths, n, k, max_delta_rot, speed_range, x_init=None, y_init=None, xy_range=xy_range, save_path=save_path)

#x = input_x[:500, 0].T.tolist()
#y = input_x[:500, 1].T.tolist()

#print(input_x[:,:])
#print(y)

plt.scatter(x, y, s=0.1, c='purple')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()


# 2d neuron interaction mask, 100 neurons
dim = 10

