# Daniel Mishler
# Last push to github 2022-06-08

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
# The last globals are
# `tasks_at_frame`
# This is an array of the number of tasks which have executed since the previous
# frame.
def animate_trace(trace, 
                  trace_type,
                  order_func = None,
                  which_animate = "tasks",
                  title = "unknown desc (pass something as 'title')",
                  num_frames = 50,         # Number of frames to use in the animation
                  enforce_interval = None, # Alternatively, determines a timestep per frame (in seconds)
                  fps = 13,
                  M=4000,
                  N=4000,
                  K=4000,
                  tilesize = 200):
    
    global A_status
    global B_status
    global C_status
    global tasks_at_frame
    start_time = time.time()
    print("Beginning animation of data '%s' method '%s'" % (title, which_animate))
    
    # Begin checks
    if trace_type == "ptg": # TODO: find this programmatically
        gemm_index = 17
    elif trace_type == "dtd":
        gemm_index = 19
        if order_func is None:
            print("Error: for DTD traces, you must supply an order function!")
            return
    else:
        print("Error: unknown trace type '" + str(trace_type) + "'")
        return
    
    # File type checks
    if (len(trace.events[trace.events.type == gemm_index]["id"].unique()) !=
       len(trace.events[trace.events.type == gemm_index]["id"])):
        print("Error: file is likely of invalid type")
        return
    
    if which_animate not in ["tasks", "abctasks", "progress", "abcprogress"]:
        print("warning, I don't know what you wanted me to animate (which_animate='%s')" % which_animate)
        print("I'll animate the progress animation")
        which_animate = "progress"
    # End checks
    # Begin estimation and preprocessing
    orderdf = pd.DataFrame(trace.events[trace.events.type == gemm_index].sort_values("begin"))
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
    Ntiles_m = math.ceil(M/tilesize)
    Ntiles_n = math.ceil(N/tilesize)
    Ntiles_k = math.ceil(K/tilesize)
    # Check to see if we have as many tasks as we should expect for a GEMM
    if(Ntiles_m * Ntiles_n * Ntiles_k != len(orderdf)):
        print("warning: it seems like your trace has %d tasks, but I expected %d "
                 % (len(orderdf), Ntiles_m * Ntiles_n * Ntiles_k)
                 + "based on the arguments you provided for N, M, K, and tilesize.")
    
    
    # This ID array is only needed for dtd traces.
    # (Because the trace for DTD gemm does not supply us with
    # m, n, and k, we must remember the order we insterted the tasks
    # and re-extract that data as such:
    if trace_type == "dtd":
        id_orders = np.array(orderdf["id"])
        
        # Prepare an list of indices that index which tasks were executing
        # based on the list of the order the tasks were inserted
        indices_arr = np.zeros(len(id_orders),dtype=int)
        check_for = 0
        for id_normed in range(len(indices_arr)):
            while(len(np.where(id_orders == check_for)[0]) == 0):
                check_for += 1
            indices_arr[np.where(id_orders == check_for)[0][0]] = id_normed
            check_for += 1
        # Set up the ideal order given the implementation provided
        ideal_order = order_func(Ntiles_m, Ntiles_n, Ntiles_k)
    
    
    C_status = []
    for i in range(Ntiles_m):
        C_status.append(np.zeros(Ntiles_n))
    C_status = np.array(C_status)
    
    if(which_animate == "abcprogress" or which_animate == "abctasks"):
        A_status = []
        for i in range(Ntiles_m):
            A_status.append(np.zeros(Ntiles_k))
        A_status = np.array(A_status)
        B_status = []
        for i in range(Ntiles_k):
            B_status.append(np.zeros(Ntiles_n))
        B_status = np.array(B_status)
    
    
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
    # mode: 'progress' for keeping the tiles colored, 'tasks' to reset them and see what was active and when.
    # FuncAnimate works best with an init function, even if it doesn't do much or the init is done elsewhere.
    def animate_init_common(plots, mode):
        global A_status
        global B_status
        global C_status
        global tasks_at_frame
        tasks_at_frame = []
        C_status = C_status * 0
        if(plots == "abc"):
            A_status = A_status * 0
            B_status = B_status * 0
        
        return
    
    def animate_init_with_time():
        animate_init_common("c", "tasks")
    
    def animate_init_with_time_all():
        animate_init_common("abc", "tasks")
    
    def animate_init_progress():
        animate_init_common("c", "progress")
    
    def animate_init_progress_all():
        animate_init_common("abc", "progress")
    
    # for the actual animation
    def animate_common(frame, trace_type, plots, mode):
        global A_status
        global B_status
        global C_status
        global tasks_at_frame
        # time_point_curr and time_point_prev in units of nanoseconds
        time_point_curr = ((frame+1)*(last_end - first_begin))//num_frames + first_begin
        time_point_prev = ((frame+0)*(last_end - first_begin))//num_frames + first_begin
        if(frame >= num_frames): # This construction to allow dead frames at the end
            time_point_prev = time_point_curr = last_end
            
        if(mode == "tasks"):
            C_status = C_status * 0 # Always zero when doing task timing
            if(plots == "abc"):
                A_status = A_status * 0 # Always zero when doing task timing
                B_status = B_status * 0 # Always zero when doing task timing
            
        tasks_before = orderdf.loc[orderdf["end"] <= time_point_curr]
        tasks_during = tasks_before.loc[(tasks_before["end"] > time_point_prev)]
            # aside: you might think this construction with > time_point_prev wouldn't include the first task,
            #        but this is fine! The only assumption made is that the first tasks does not end the same nanoseond
            #        that it begins. That sounds like a safe assumption to me.
        
        tasks_at_frame.append(len(tasks_during))
        
        if trace_type == "dtd":
            for tid in tasks_during["id"]:
                tid_normed = indices_arr[np.where(id_orders == tid)][0]
                element = ideal_order[indices_arr[tid_normed]]
                if(plots == "c"):
                    C_status[element[0],element[1]] += 1
                elif(plots == "abc"):
                    A_status[element[0],element[2]] += 1
                    B_status[element[2],element[1]] += 1
                    C_status[element[0],element[1]] += 1
        elif trace_type == "ptg":
            for index, task in tasks_during.iterrows():
                if(plots == "c"):
                    C_status[task["m"], task["n"]] += 1
                elif(plots == "abc"):
                    A_status[task["m"], task["k"]] += 1
                    B_status[task["k"], task["n"]] += 1
                    C_status[task["m"], task["n"]] += 1
                
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
        return
    
    def animate_dtd_order_with_time(frame):
        animate_common(frame, "dtd", "c", "tasks")
        return
    
    def animate_dtd_order_with_time_all(frame):
        animate_common(frame, "dtd", "abc", "tasks")
        return
    
    def animate_dtd_order_progress(frame):
        animate_common(frame, "dtd", "c", "progress")
        return
        
    def animate_dtd_order_progress_all(frame):
        animate_common(frame, "dtd", "abc", "progress")
        return
    
    def animate_ptg_order_with_time(frame):
        animate_common(frame, "ptg", "c", "tasks")
        return
    
    def animate_ptg_order_with_time_all(frame):
        animate_common(frame, "ptg", "abc", "tasks")
        return
    
    def animate_ptg_order_progress(frame):
        animate_common(frame, "ptg", "c", "progress")
        return
        
    def animate_ptg_order_progress_all(frame):
        animate_common(frame, "ptg", "abc", "progress")
        return
    
    if(trace_type == "dtd"):
        if(which_animate == "tasks"):
            animation_func = animate_dtd_order_with_time
            animation_init = animate_init_with_time
        elif(which_animate == "progress"):
            animation_func = animate_dtd_order_progress
            animation_init = animate_init_progress
        elif(which_animate == "abctasks"):
            animation_func = animate_dtd_order_with_time_all
            animation_init = animate_init_with_time_all
        elif(which_animate == "abcprogress"):
            animation_func = animate_dtd_order_progress_all
            animation_init = animate_init_progress_all
    elif(trace_type == "ptg"):
        if(which_animate == "tasks"):
            animation_func = animate_ptg_order_with_time
            animation_init = animate_init_with_time
        elif(which_animate == "progress"):
            animation_func = animate_ptg_order_progress
            animation_init = animate_init_progress
        elif(which_animate == "abctasks"):
            animation_func = animate_ptg_order_with_time_all
            animation_init = animate_init_with_time_all
        elif(which_animate == "abcprogress"):
            animation_func = animate_ptg_order_progress_all
            animation_init = animate_init_progress_all
        
    padding_frames = fps // 2 # Supply a half second of stillness at the end
    animation_result = FuncAnimation(fig,animation_func, init_func = animation_init,
                                     frames=(num_frames+padding_frames),interval=1000//fps)
    video = animation_result.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()
    

    end_time = time.time()
    ## Printing
    tasks_times = (orderdf["end"] - orderdf["begin"]).to_numpy()
    tasks_times = tasks_times / 1000000 # convert from ns to ms
    tasks_execution_mean = tasks_times.mean()
    tasks_execution_sdev = tasks_times.std()
    tasks_runtime = (last_end - first_begin) / 1000000 # in ms
    utilization = (tasks_execution_mean * len(tasks_times)) / tasks_runtime
    num_cores = trace.information["nb_cores"]
    print("Data titled '%s'" % title)
    print("M=%d,\tN=%d,\tK=%d,\ttilesize=%d" % (M,N,K,tilesize))
    print("average task execution time: %f" % tasks_execution_mean)
    print("task execution time standard deviation: %f" % tasks_execution_sdev)
    print("utilization: %f over %d cores (%f)" % (utilization, num_cores, utilization/num_cores))
    print("execution time to generate graphs: %f seconds" % (end_time-start_time))
    
    # Plot of the file tasks per frame
    fig, ax = plt.subplots(1, figsize = [5,5])
    times_array = np.array(range(num_frames+padding_frames))*time_per_frame*1000
    ax.plot(times_array, tasks_at_frame, "*-")
    ax.set_title("tasks since each frame (%s)" % title)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("number of tasks")
    fname = "tasks_per_frame_(%s).png" % title
    plt.savefig(fname)
    # Offer the user the filename in case they wish to display it after
    print("saved task metadata file:", fname)
    
    
    # Plot of the task timings
    # currently sorted by task beginning time
    fig, ax = plt.subplots(1, figsize = [5,5])
    ax.plot(range(len(tasks_times)), tasks_times, "*")
    ax.set_title("Timing of Each Task (%s)" % title)
    ax.set_xlabel("task number (sorted by beginning time)")
    ax.set_ylabel("task duration (ms)")
    fname = "tasks_times_execution_order_(%s).png" % title
    plt.savefig(fname)
    # Offer the user the filename in case they wish to display it after
    print("saved task metadata file:", fname)
    
    
    # Do it again but sort them by execution duration
    tasks_times.sort() # Now sorted by task duration
    fig, ax = plt.subplots(1, figsize = [5,5])
    ax.plot(range(len(tasks_times)), tasks_times, "*")
    ax.set_title("Timing of Each Task (%s)" % title)
    ax.set_xlabel("task number (sorted by beginning time)")
    ax.set_ylabel("task duration (ms)")
    fname = "tasks_times_sorted_(%s).png" % title
    plt.savefig(fname)
    # Offer the user the filename in case they wish to display it after
    print("saved task metadata file:", fname)
    
    if(sum(tasks_at_frame) != Ntiles_m * Ntiles_n * Ntiles_k):
        print("error: expected %d tasks but I only observed %d in the trace" %
              (Ntiles_m * Ntiles_n * Ntiles_k, sum(tasks_at_frame)))
    
    print()

    
    
    