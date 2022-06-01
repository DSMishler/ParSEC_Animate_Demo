# Daniel Mishler
# Last push to github 2022-06-01

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import numpy as np
import math
import time


'''
What the deal with order generators?
When using task insertion through dtd, the trace does not track data such as the m, n, and k tiles.
Because of this, we need to take one extra step to know what order the tasks were generated in
(since they are monotonically sorted by id) to extract this information from the trace.
'''

def rectangle_order_generator_improved(Ntiles_m, Ntiles_n=None, Ntiles_k=None, square_size = 6):
    # assumes square matrix which takes N tiles to cover
    i = 0 # parsing through m
    j = 0 # parsing through n
    k = 0
    if Ntiles_n is None:
        Ntiles_n = Ntiles_m
    if Ntiles_k is None:
        Ntiles_k = Ntiles_m
    checkpoint_i = 0
    checkpoint_j = 0
    ordered_list = []
    while(checkpoint_i < Ntiles_m):
        checkpoint_j = 0
        while(checkpoint_j < Ntiles_n):
            k = 0
            while(k < Ntiles_k):
                i = checkpoint_i
                while(i < checkpoint_i + square_size and i < Ntiles_m):
                    j = checkpoint_j
                    while(j < checkpoint_j + square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k += 1
            k = Ntiles_k - 1 #k--
            while(k >= 0):
                i = checkpoint_i+square_size
                while(i < checkpoint_i + 2*square_size and i < Ntiles_m):
                    j = checkpoint_j
                    while(j < checkpoint_j + square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k -= 1
            k = 0
            while(k < Ntiles_k):
                i = checkpoint_i+square_size
                while(i < checkpoint_i + 2*square_size and i < Ntiles_m):
                    j = checkpoint_j + square_size
                    while(j < checkpoint_j + 2*square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k += 1
            k = Ntiles_k - 1 #k--
            while(k >= 0):
                i = checkpoint_i
                while(i < checkpoint_i + square_size and i < Ntiles_m):
                    j = checkpoint_j + square_size
                    while(j < checkpoint_j + 2*square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k -= 1
            checkpoint_j += (2 * square_size)
        checkpoint_i += (2 * square_size)
    return ordered_list

def rectangle_order_generator_naive(Ntiles_m, Ntiles_n=None, Ntiles_k=None, square_size = 6):
    # assumes square matrix which takes N tiles to cover
    i = 0 # parsing through m
    j = 0 # parsing through n
    k = 0
    if Ntiles_n is None:
        Ntiles_n = Ntiles_m
    if Ntiles_k is None:
        Ntiles_k = Ntiles_m
    checkpoint_i = 0
    checkpoint_j = 0
    ordered_list = []
    while(checkpoint_i < Ntiles_m):
        checkpoint_j = 0
        while(checkpoint_j < Ntiles_n):
            k = 0
            while(k < Ntiles_k):
                i = checkpoint_i
                while(i < checkpoint_i + square_size and i < Ntiles_m):
                    j = checkpoint_j
                    while(j < checkpoint_j + square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k += 1
            k = Ntiles_k - 1 #k--
            while(k >= 0):
                i = checkpoint_i+square_size
                while(i < checkpoint_i + 2*square_size and i < Ntiles_m):
                    j = checkpoint_j
                    while(j < checkpoint_j + square_size and j < Ntiles_n):
                        ordered_list.append([i,j,k])
                        j += 1
                    i += 1
                k -= 1
            checkpoint_j += square_size
        checkpoint_i += (2 * square_size)
    return ordered_list


def sequential_order_generator(Ntiles_m, Ntiles_n, Ntiles_k, square_size = 6):
    i = 0 # parsing through m
    j = 0 # parsing through n
    k = 0
    ordered_list = []
    while(i < Ntiles_m):
        j = 0
        while(j < Ntiles_n):
            k = 0
            while(k < Ntiles_k):
                ordered_list.append([i,j,k])
                k += 1
            j += 1
        i += 1
        
    return ordered_list

    
# Some globals are needed for the below function:
# A_status, B_status, C_status
# These lists of lists represent the progress of the job on each tile
# Such lists are modified both by animate_trace and a function within it,
# which is necessary to implement in order to achieve the format that 
# is required by FuncAnimation
# The fourth global is
# tasks_at_frame
# This is an array of the number of tasks which have executed since the previous
# frame.
def animate_trace(trace, order_func,
                  which_animate = "tasks",
                  title = "unknown desc (pass something as 'title')",
                  num_frames = 50,         # Number of frames to use in the animation
                  enforce_interval = None, # Alternatively, determines a timestep per frame (in seconds)
                  fps = 13,
                  M=4000,
                  N=4000,
                  K=4000,
                  tilesize = 200):
    start_time = time.time()
    print("Beginning animation of data '%s' method '%s'" % (title, which_animate))
    
    # Begin checks
    
    # File type checks
    if len(trace.events[trace.events.type == 19]["id"].unique()) != len(trace.events[trace.events.type == 19]["id"]):
        print("Error: file is likely of invalid type")
        return
    
    if which_animate not in ["tasks", "abctasks", "progress", "abcprogress"]:
        print("warning, I don't know what you wanted me to animate (which_animate='%s')" % which_animate)
        print("I'll animate the progress animation")
        which_animate = "progress"
    # End checks
    # Begin estimation and preprocessing
    orderdf = pd.DataFrame(trace.events[trace.events.type == 19].sort_values("begin"))
    first_end = orderdf["end"].iloc[0]
    first_begin = orderdf["begin"].iloc[0]
    last_end =  orderdf["end"].iloc[-1]
    
    # Possibly enforce an interval in seconds
    if enforce_interval is not None:
        time_per_frame = enforce_interval
        num_frames = math.ceil((last_end - first_begin)/10**9/time_per_frame)
        print("enforcing %d frames to grant you your requested enforced interval" % (num_frames))
    time_per_frame = (last_end - first_begin)/10**9/num_frames
    print("process runtime per frame: %f seconds" % (time_per_frame))
    
    estimation_multiplier = 3 # for abc type calls
    print("estimated execution time (assuming a lightweight commercial processor): %f seconds" % (num_frames * 0.3 * estimation_multiplier))
    if(num_frames > 100):
        print("animate_trace warning: num_frames beyond 100 may cause long compute time")
    if(num_frames > 500):
        print("animate_trace warning: num_frames beyond 500 may cause extreme compute time")
    if(num_frames <= 0):
        print("Error: illegal number of frames. Must be at least 1")
        return
    
    # End estimation and preprocessing
    id_orders = np.array(orderdf["id"])
    
    # Prepare an list of indices that index which tasks were executing based on the list of the order the tasks were inserted
    indices_arr = np.zeros(len(id_orders),dtype=int)
    check_for = 0
    for id_normed in range(len(indices_arr)):
        while(len(np.where(id_orders == check_for)[0]) == 0):
            check_for += 1
        indices_arr[np.where(id_orders == check_for)[0][0]] = id_normed
        check_for += 1
    
    Ntiles_m = math.ceil(M/tilesize)
    Ntiles_n = math.ceil(N/tilesize)
    Ntiles_k = math.ceil(K/tilesize)
    global A_status
    global B_status
    global C_status
    global tasks_at_frame
    A_status = []
    B_status = []
    C_status = []
    for i in range(Ntiles_m):
        A_status.append(np.zeros(Ntiles_k))
    for i in range(Ntiles_k):
        B_status.append(np.zeros(Ntiles_n))
    for i in range(Ntiles_m):
        C_status.append(np.zeros(Ntiles_n))
    A_status = np.array(A_status)
    B_status = np.array(B_status)
    C_status = np.array(C_status)
    ideal_order = order_func(Ntiles_m, Ntiles_n, Ntiles_k)
    
    # Prepare the figure that will be displayed
    fld = 6.5 # figsize larger dimension
    fmd = 3.5 # figsize minimum dimension
    if Ntiles_m >= Ntiles_n:
        figsize_x = fld*Ntiles_n/Ntiles_m
        figsize_y = fld
        if(figsize_x < fmd):
            figsize_x = fmd
    else:
        figsize_x = fld
        figsize_y = fld*Ntiles_m/Ntiles_n
        if(figsize_y < fmd):
            figsize_y = fmd
            
    visual_vmax = Ntiles_k
    # Make the figure
    if(which_animate == "abcprogress" or which_animate == "abctasks"):
        fig, ((axx, axB), (axA, axC)) = plt.subplots(2,2, figsize = [figsize_x, figsize_y])
        # fig.colorbar(axC.pcolor(C_status,vmin=0,vmax=visual_vmax))
        axA.pcolor(A_status, vmin = 0, vmax = Ntiles_n)
        axA.invert_yaxis()
        axB.pcolor(B_status, vmin = 0, vmax = Ntiles_m)
        axB.invert_yaxis()
        axC.pcolor(C_status, vmin = 0, vmax = Ntiles_k)
        axC.invert_yaxis()
    else:
        fig, ax = plt.subplots(1, figsize = [figsize_x, figsize_y])
        ax.pcolor(C_status, vmin = 0, vmax = visual_vmax)
        fig.colorbar(ax.pcolor(C_status,vmin=0,vmax=visual_vmax))
        ax.invert_yaxis()
    
    # Enter the animation functions
    
    # plots: 'c' for just plotting matrix c, 'abc' for all three
    # mode: 'progress' for keeping the trace active, 'tasks' to see what was active and when.
    def dtd_animate_common(frame, plots, mode):
        global A_status
        global B_status
        global C_status
        global tasks_at_frame
        time_point_curr = int(frame*(last_end - first_begin)/num_frames) + first_begin
        time_point_prev = int((frame-1)*(last_end - first_begin)/num_frames) + first_begin
        if(mode == "tasks"):
            C_status = C_status * 0 # Always zero when doing task timing
            if(plots == "abc"):
                A_status = A_status * 0 # Always zero when doing task timing
                B_status = B_status * 0 # Always zero when doing task timing
        if(frame == 0):
            tasks_at_frame = []
            C_status = C_status * 0
            if(plots == "abc"):
                A_status = A_status * 0
                B_status = B_status * 0
        if(frame == num_frames - 1):
            time_point_curr = last_end
        if(frame >= num_frames):
            time_point_prev = time_point_curr = last_end
            
            
        tasks_before = orderdf.loc[orderdf["end"] <= time_point_curr]
        tasks_during = tasks_before.loc[(tasks_before["end"] > time_point_prev)]
        
        tasks_at_frame.append(len(tasks_during))
        # for task in tasks_during:
            # pass
        
        for tid in tasks_during["id"]:
            tid_normed = indices_arr[np.where(id_orders == tid)][0]
            element = ideal_order[indices_arr[tid_normed]]
            if(plots == "c"):
                C_status[element[0],element[1]] += 1
            elif(plots == "abc"):
                A_status[element[0],element[2]] += 1
                B_status[element[2],element[1]] += 1
                C_status[element[0],element[1]] += 1
        vmax_A = Ntiles_n
        vmax_B = Ntiles_m
        vmax_C = Ntiles_k
        if(plots == "c"):
            ax.set_title("Matrix C at time t=%fs (%s)" % (time_point_curr/10**9, title))
            ax.pcolor(C_status, vmin = 0, vmax = vmax_C)
        elif(plots == "abc"):
            axA.set_title("Matrix A")
            axA.pcolor(A_status, vmin = 0, vmax = vmax_A)
            axB.set_title("Matrix B")
            axB.pcolor(B_status, vmin = 0, vmax = vmax_B)
            axC.set_title("Matrix C")
            axC.pcolor(C_status, vmin = 0, vmax = vmax_C)
    
    def animate_order_with_time(frame):
        dtd_animate_common(frame, "c", "tasks")
        return
    
    def animate_order_with_time_all(frame):
        dtd_animate_common(frame, "abc", "tasks")
        return
    
    def animate_order_progress(frame):
        dtd_animate_common(frame, "c", "progress")
        return
        
    def animate_order_progress_all(frame):
        dtd_animate_common(frame, "abc", "progress")
        return
    
    if(which_animate == "tasks"):
        animation_func = animate_order_with_time
    elif(which_animate == "progress"):
        animation_func = animate_order_progress
    elif(which_animate == "abctasks"):
        animation_func = animate_order_with_time_all
    elif(which_animate == "abcprogress"):
        animation_func = animate_order_progress_all
        
    padding_frames = fps // 2 # Supply a half second of stillness at the end
    animation_result = FuncAnimation(fig,animation_func,frames=(num_frames+padding_frames),interval=int(1000/fps))
    video = animation_result.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()
    

    end_time = time.time()
    ## Printing
    print("Data titled '%s'" % title)
    print("M=%d\tN=%d\tK=%d" % (M,N,K))
    print("execution time to generate graphs: %f seconds" % (end_time-start_time))
    
    # Plot of the file tasks per frame
    fig, ax = plt.subplots(1, figsize = [5,5])
    ax.plot(range(num_frames+padding_frames), tasks_at_frame)
    ax.set_title("tasks since each frame")
    ax.set_xlabel("frame number")
    ax.set_ylabel("number of tasks")
    fname = "tasks_per_frame_(%s).png"%title
    # Offer the user the filename in case they wish to display it after
    print("saved task metadata file:", fname)
    plt.savefig(fname)
    
    print()

    
    
    