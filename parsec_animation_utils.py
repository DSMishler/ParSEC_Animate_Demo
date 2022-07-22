# Daniel Mishler
# Last push to github 2022-07-22

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



################################################################################################
# helper functions to make the animation function itself shorter and more readable
################################################################################################
def guess_trace_type(trace):
    # TODO: Make a more sophisticated guess
    try:
        trace.events["m"]
        trace_type = "ptg"
    except KeyError:
        trace_type = "dtd"
    return trace_type

def guess_running_system(trace):
    # TODO: Make a more sophisticated guess
    try:
        trace.information["HWLOC-XML"]
        running_system = "dplasma"
    except KeyError:
        running_system = "hicma"
    return running_system

def check_for_order_function(trace_type, order_func):
    if trace_type == "ptg":
        error = False
    elif trace_type == "dtd":
        if order_func is None:
            print("Error: for DTD traces, you must supply an order function!")
            error = True
        else:
            error = False
    else:
        print("Error: unknown trace type '" + str(trace_type) + "'")
        error = True
    return error

def check_parameter_compatibility(task_type, trace_type, which_animate, M, N, K):
    if task_type == "gemm":
        error = False
    elif task_type == "potrf":
        error = False
        if trace_type != "ptg":
            print("Error: potrf only supported with ptg")
            error = True
        if(which_animate == "abcprogress" or which_animate == "abctasks"):
            print("Error: potrf only supported with C view")
            error = True
        if N != M:
            print("Error: only square animations of potrf supported")
            error = True
    else:
        print("Error: unknown task type '" + str(task_type) +"'")
        error = True
    return error
    
def get_work_tasks_indices(trace, task_type, running_system):
    # TODO: Feels like this one could use some generalization and cleanup too.
    # TODO: split this into two functions, each with 1 return
    work_tasks_indices = []
    if task_type == "gemm":
        name_to_task_num = None
        gemm_index_found = 0
        for name in trace.event_types.index: # An array of numbers indexed by their name. Interesting, huh?
            if name.lower() == "gemm":
                work_tasks_indices.append(trace.event_types[name])
                gemm_index_found += 1
                break
        if gemm_index_found != 1:
            print("Error: file trace does not have its event_types set properly")
            print("found %d events, expected %d" % (gemm_index_found, 1))
            return None, None
    
    elif task_type == "potrf":
        name_to_task_num = {}
        potrf_index_found = 0
        for name in trace.event_types.index:
            if "potrf_" in name.lower(): # Could possibly use regex for this...
                for taskname in ["gemm", "syrk", "trsm", "potrf"]:
                    if(taskname in name.split('_')[-1].lower()):
                        print("found task type %s (#%d)" % (name.lower(), trace.event_types[name]))
                        work_tasks_indices.append(trace.event_types[name])
                        if(running_system == "hicma"):
                            if("3flow" in name.lower()):
                                name_to_task_num["large-"+taskname] = trace.event_types[name]
                            else:
                                name_to_task_num["small-"+taskname] = trace.event_types[name]
                        elif(running_system == "dplasma"):
                            name_to_task_num[taskname] = trace.event_types[name]
                        else:
                            print("error: unknown running system")
                        potrf_index_found += 1
                        break
        if(running_system == "hicma"):
            expected_task_types = 8
        elif(running_system == "dplasma"):
            expected_task_types = 4
        if potrf_index_found != expected_task_types:
            print("Error: file trace does not have its event_types set properly")
            print("found %d events, expected %d" % (potrf_index_found, 4))
            return None, None
    return (work_tasks_indices, name_to_task_num)
    
def get_potrf_uplo(running_system, orderdf, name_to_task_num):
    # TODO: There gotta be a better way to do this than peek at a task, no?
    if running_system == "dplasma":
        test_task = "trsm"
    elif running_system == "hicma":
        test_task = "small-trsm"
    else:
        print("Error, unknown running system!")
        return None
    
    if orderdf.loc[orderdf["type"] == name_to_task_num[test_task]].iloc[0]["m"] is None:
        potrf_uplo = "upper"
    else:
        potrf_uplo = "lower"
    return potrf_uplo

def get_time_per_frame(first_begin, last_end, num_frames, enforce_interval):
    # Possibly enforce an interval in seconds
    if enforce_interval is not None:
        time_per_frame = enforce_interval
        num_frames = math.ceil((last_end - first_begin)/10**9/time_per_frame)
        print("enforcing %d frames to grant you your requested enforced interval" % (num_frames))
    if(num_frames <= 0):
        print("Error: illegal number of frames. Must be at least 1")
        return None
    time_per_frame = (last_end - first_begin)/10**9/num_frames
    return time_per_frame

def get_estimated_video_time(which_animate, running_system, task_type, num_frames, Ntiles_m, Ntiles_n):
    # Estimate time to get graphs
    # TODO: find a better way to calculate this - currently it's just loosely
    #       fit to whatever I observe on my laptop.
    estimation_multiplier = 1
    if(which_animate == "abcprogress" or which_animate == "abctasks"):
        estimation_multiplier *= 2 # for abc type calls
    if running_system == "hicma":
        estimation_multiplier *= 0.20
    if task_type == "gemm":
        estimation_multiplier *= 3
    estimated_compile_time = (num_frames * 0.0004 * ((Ntiles_m * Ntiles_n)**0.9) * estimation_multiplier)
    return estimated_compile_time

def get_expected_tasks(task_type, running_system, Ntiles_m, Ntiles_n, Ntiles_k):
    # How many work tasks should I expect, given what I know about the problem?
    # This information will prove useful for error checking.
    if task_type == "gemm":
        expected_tasks = Ntiles_m * Ntiles_n * Ntiles_k
    elif task_type == "potrf":
        if Ntiles_m != Ntiles_n:
            print("error: expected 'n' and 'm' to be equal to POTRF trace")
            return None
        # expected_tasks = Ntiles_m # potrf
        # expected_tasks += ((Ntiles_m-1)*(Ntiles_m)) // 2 # trsm
        # expected_tasks += ((Ntiles_m-1)*(Ntiles_m)) // 2 # syrk
        # for i in range(Ntiles_m-1): # gemm 
            # expected_tasks += (i*(i+1)) // 2
        # Alternative methods
        # n * 1 + (n-1) * 2 + (n-2) * 3 + (n-3) * 4 + ...
        # which is same as: sum over (x^2/2 + x/2) from 1 to n
        # which is the same as: (n) * (n+1) * (n+2) / 6
        if running_system == "dplasma":
            expected_tasks = (Ntiles_m)*(Ntiles_m+1)*(Ntiles_m+2)//6
        elif running_system == "hicma":
            if bigtilesize % tilesize != 0:
                print("error: cannot have large tilesize and smaller tilesize not evenly divide.")
                return None
            small_tiles = bigtilesize//tilesize
            Nbigtiles = math.ceil(M/bigtilesize)
            # large tasks
            expected_tasks = (Nbigtiles)*(Nbigtiles+1)*(Nbigtiles+2)//6
            # small tasks
            tasks_per_potrf = (small_tiles)*(small_tiles+1)*(small_tiles+2)//6
            expected_tasks += Nbigtiles * (tasks_per_potrf)
    return expected_tasks

def get_id_orders(orderdf, order_func, Ntiles_m, Ntiles_n, Ntiles_k):
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
    return (id_orders, ideal_order, indices_arr)
    
def init_global_arrays(Ntiles_m, Ntiles_n, Ntiles_k, which_animate):
    global A_status
    global B_status
    global C_status
    global C_stream_id
    global A_expected # Currently unused
    global B_expected
    global C_expected
    C_status = []
    C_expected = []
    C_stream_id = []
    for i in range(Ntiles_m):
        C_status.append(np.zeros(Ntiles_n))
        C_expected.append(np.zeros(Ntiles_n))
        C_stream_id.append(np.zeros(Ntiles_n))
    C_status = np.array(C_status)
    C_expected = np.array(C_expected)
    C_stream_id = np.array(C_stream_id)
    
    if(which_animate == "abcprogress" or which_animate == "abctasks"):
        A_status = []
        A_expected = []
        for i in range(Ntiles_m):
            A_status.append(np.zeros(Ntiles_k))
            A_expected.append(np.zeros(Ntiles_k))
        A_status = np.array(A_status)
        A_expected = np.array(A_expected)
        B_status = []
        B_expected = []
        for i in range(Ntiles_k):
            B_status.append(np.zeros(Ntiles_n))
            B_expected.append(np.zeros(Ntiles_n))
        B_status = np.array(B_status)
        B_expected = np.array(B_expected)
    
def get_figsize_dimensions(Ntiles_m, Ntiles_n):
    # TODO: get this to conside Ntiles_k
    fmxd = 6.5 # figsize maximum dimension
    fmnd = 3.5 # figsize minimum dimension
    if Ntiles_m >= Ntiles_n:
        figsize_x = fmxd*Ntiles_n/Ntiles_m
        figsize_y = fmxd
        if(figsize_x < fmnd):
            figsize_x = fmnd
    else:
        figsize_x = fmxd
        figsize_y = fmxd*Ntiles_m/Ntiles_n
        if(figsize_y < fmnd):
            figsize_y = fmnd
    figsize_dims = [figsize_x, figsize_y]
    return figsize_dims

def init_plt_figure(which_animate, figsize_dims):
    if(which_animate == "abcprogress" or which_animate == "abctasks"):
        fig, ((axx, axB), (axA, axC)) = plt.subplots(2,2, figsize = figsize_dims)
        # fig.colorbar(axC.pcolormesh(C_status,vmin=0,vmax=1))
        axA.pcolormesh(A_status, vmin = 0, vmax = 1)
        axA.invert_yaxis()
        axB.pcolormesh(B_status, vmin = 0, vmax = 1)
        axB.invert_yaxis()
        axC.pcolormesh(C_status, vmin = 0, vmax = 1)
        axC.invert_yaxis()
    else:
        fig, axC = plt.subplots(1, figsize = figsize_dims)
        axC.pcolormesh(C_status, vmin = 0, vmax = 1)
        # fig.colorbar(axC.pcolormesh(C_status,vmin=0,vmax=1))
        axC.invert_yaxis()
        axx = None
        axA = None
        axB = None
    return (fig, ((axx, axB), (axA, axC)))
################################################################################################
# trace function
################################################################################################
# Some globals are needed for the below function:
# A_status, B_status, C_status, A_expected, B_expected, C_expected
# These lists of lists represent the progress of the job on each tile
# Such lists are modified both by animate_trace and a function within it,
# which is necessary to implement in order to achieve the format that 
# is required by FuncAnimation.
# The `expected` variables are so that calculating progress as a percentage
# is feasible.
# The last globals are
# `tasks_at_frame`
# This is an array of the number of tasks which have executed since the previous
# frame.
# `C_stream_id`
# An array, much like C_status, which contains that last `stream_id` of the core
# that ran the last task in that location

def animate_trace(trace, 
                  task_type,
                  order_func = None,
                  which_animate = "tasks",
                  title = "unknown desc (pass something as 'title')",
                  num_frames = 50,         # Number of frames to use in the animation
                  enforce_interval = None, # Alternatively, determines a timestep per frame (in seconds)
                  fps = 13,
                  fill = "relative", # relative for all will be yellow, absolute for some will be fully yellow
                  M = None, # Must be provided
                  N = None, # Must be provided
                  K = None, # Must be provided (even if the geometry of the matrix doesn't make sense for it)
                  tilesize = None, # Must be provided
                  bigtilesize = None # Optional
                 ):
    
    global A_status
    global B_status
    global C_status
    global tasks_at_frame
    global C_stream_id
    global A_expected # Currently unimplemented
    global B_expected
    global C_expected
    start_time = time.time()
    print("Beginning animation of data '%s' method '%s'" % (title, which_animate))
    
    # Begin checks
    if M is None:
        print("Error: must provide argument for M")
        return
    if N is None:
        print("Error: must provide argument for N")
        return
    if K is None:
        print("Error: must provide argument for K (even if you don't think it makes sense in the problem!)")
        return
    if tilesize is None:
        print("Error: must provide argument for tilesize (even if you don't think it makes sense in the problem!)")
        return
    
    if which_animate not in ["tasks", "abctasks", "progress", "abcprogress", "coreswaps"]:
        print("warning, I don't know what you wanted me to animate (which_animate='%s')" % which_animate)
        print("I'll animate the progress animation")
        which_animate = "progress"
    
    legal_fills = ["relative", "absolute"]
    if fill not in legal_fills:
        print("Error: fill must be in ", legal_fills)
        return
        
    
    # Guess the trace type
    trace_type = guess_trace_type(trace)
    print("After observing your trace, I am guessing it is from the %s interface" % trace_type)
    
    running_system = guess_running_system(trace)
    print("I think this trace is for a job that was running on", running_system)
    
    error = check_for_order_function(trace_type, order_func)
    if error == True:
        return
    
    
    error = check_parameter_compatibility(task_type, trace_type, which_animate, M, N, K)
    if error == True:
        return
    
    
    # Begin estimation and preprocessing
    (work_tasks_indices, name_to_task_num) = get_work_tasks_indices(trace, task_type, running_system)
    if work_tasks_indices is None or (name_to_task_num is None and trace_type == "potrf"):
        return
    
    # Now isolate just the work tasks
    # This is the step that seems to take forever.
    work_tasks = trace.events.type.isin(work_tasks_indices)
    orderdf = pd.DataFrame(trace.events[work_tasks].sort_values("begin"))
    
    if task_type == "potrf":
        potrf_uplo = get_potrf_uplo(running_system, orderdf, name_to_task_num)
        if potrf_uplo is None:
            return
        print("I think this is potrf is %s triangular" % potrf_uplo)
        
    first_begin = orderdf["begin"].iloc[0]
    last_end =  orderdf["end"].iloc[-1]
    
    Ntiles_m = math.ceil(M/tilesize)
    Ntiles_n = math.ceil(N/tilesize)
    Ntiles_k = math.ceil(K/tilesize)
    
    
    time_per_frame = get_time_per_frame(first_begin, last_end, num_frames, enforce_interval)
    if time_per_frame is None:
        return
    print("process runtime per frame: %f seconds" % (time_per_frame))
    
    
    estimated_compile_time = get_estimated_video_time(which_animate, running_system,
                                                      task_type, num_frames,
                                                      Ntiles_m, Ntiles_n)
    print("estimated video compile time (assuming a lightweight commercial processor): %f seconds" %
          estimated_compile_time)
    
    
    expected_tasks = get_expected_tasks(task_type, running_system, Ntiles_m, Ntiles_n, Ntiles_k)
    if expected_tasks is None:
        return
    
    
    # This ID array is only needed for dtd traces.
    # (Because the trace for DTD gemm does not supply us with
    # m, n, and k, we must remember the order we insterted the tasks
    # and re-extract that data as such:
    if trace_type == "dtd":
        (id_orders, ideal_order, indices_arr) = get_id_orders(orderdf, order_func, Ntiles_m, Ntiles_n, Ntiles_k)
        if id_orders is None or ideal_order is None or indices_arr is None:
            return
    
    
    # End estimation and preprocessing
    # Prepare the arrays used for the animation
    init_global_arrays(Ntiles_m, Ntiles_n, Ntiles_k, which_animate)
    
    # Prepare the figure that will be displayed
    figsize_dims = get_figsize_dimensions(Ntiles_m, Ntiles_n)
    fig, ((axx, axB), (axA, axC)) = init_plt_figure(which_animate, figsize_dims)
    
    
    # Enter the animation functions
    # plots: 'c' for just plotting matrix c, 'abc' for all three
    # mode: 'progress' for keeping the tiles colored, 'tasks' to reset them and see what was active and when.
    
    # FuncAnimate works best with an init function, even if it doesn't do much or the init is done elsewhere.
    def animate_init_common(plots):
        global A_status
        global B_status
        global C_status
        global tasks_at_frame
        global C_stream_id
        global A_expected
        global B_expected
        global C_expected
        tasks_at_frame = []
        C_status = C_status * 0
        if(plots == "abc"):
            A_status = A_status * 0
            B_status = B_status * 0
        
        # Check to see if we have as many tasks as we should expect
        if(expected_tasks != len(orderdf)):
            print("warning: it seems like your trace has %d tasks, but I expected %d "
                     % (len(orderdf), expected_tasks)
                     + "based on the arguments you provided for N, M, K, and tilesize.")
            
        if task_type == "gemm": # it is irrelevant whether fill is relative or absolute here.
            C_expected += Ntiles_k
            if plots == "abc":
                A_expected += Ntiles_n
                B_expected += Ntiles_m
            C_stream_id -= 1 # Set stream_id regardless of the plots
        elif task_type == "potrf":
            if fill == "relative":
                for i in range(len(C_expected)):
                    for j in range(len(C_expected[0])):
                        if j <= i: # For lower triangular
                            C_expected[i,j] = j+1
                        elif j >= i: # For upper triangular
                            C_expected[i,j] = i+1
            elif fill == "absolute":
                C_expected += len(C_expected[0])
        else:
            print("Error: unknown task type")
        return
    
    def animate_init_abc():
        return animate_init_common("abc")
    
    def animate_init_c():
        return animate_init_common("c")
    
    # for the actual animation
    def animate_gemm_common(frame, trace_type, plots, mode):
        global A_status
        global B_status
        global C_status
        global tasks_at_frame
        global C_stream_id
        global A_expected
        global B_expected
        global C_expected
        # time_point_curr and time_point_prev in units of nanoseconds
        time_point_curr = ((frame+0)*(last_end - first_begin))//num_frames + first_begin
        time_point_prev = ((frame-1)*(last_end - first_begin))//num_frames + first_begin
        if(frame == 0): # One empty frame at start
            time_point_prev = time_point_cur = first_begin
        if(frame >  num_frames): # This construction to allow dead frames at the end
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
        
        for index, task in tasks_during.iterrows():
            if trace_type == "dtd":
                tid = task["id"]
                tid_normed = indices_arr[np.where(id_orders == tid)][0]
                element = ideal_order[indices_arr[tid_normed]]
                m = element[0]
                n = element[1]
                k = element[2]
                core = task["stream_id"]
            elif trace_type == "ptg":
                m = task["m"]
                n = task["n"]
                k = task["k"]
                core = task["stream_id"]
                
            if(plots == "c"):
                if mode == "swaps":
                    if(core != C_stream_id[m, n]):
                        C_stream_id[m, n] = core
                        C_status[m, n] += 1 / C_expected[m, n]
                elif mode in ["tasks", "progress"]:
                    C_status[m, n] += 1 / C_expected[m, n]
                else:
                    print("error: unknown mode", mode)
            elif(plots == "abc"):
                A_status[m, k] += 1 / A_expected[m, k]
                B_status[k, n] += 1 / B_expected[k, n]
                C_status[m, n] += 1 / C_expected[m, n]
                
        if(plots == "c"):
            axC.set_title("Matrix C at time t=%fs (%s)" % (time_point_curr/10**9, title))
            axC.pcolormesh(C_status, vmin = 0, vmax = 1)
        elif(plots == "abc"):
            axA.set_title("Matrix A")
            axA.pcolormesh(A_status, vmin = 0, vmax = 1)
            axB.set_title("Matrix B")
            axB.pcolormesh(B_status, vmin = 0, vmax = 1)
            axC.set_title("Matrix C")
            axC.pcolormesh(C_status, vmin = 0, vmax = 1)
        return
    
    def animate_gemm_dtd_order_with_time(frame):
        return animate_gemm_common(frame, "dtd", "c", "tasks")
    
    def animate_gemm_dtd_order_with_time_all(frame):
        return animate_gemm_common(frame, "dtd", "abc", "tasks")
    
    def animate_gemm_dtd_order_core_swaps(frame):
        return animate_gemm_common(frame, "dtd", "c", "swaps")
    
    def animate_gemm_dtd_order_progress(frame):
        return animate_gemm_common(frame, "dtd", "c", "progress")
        
    def animate_gemm_dtd_order_progress_all(frame):
        return animate_gemm_common(frame, "dtd", "abc", "progress")
    
    def animate_gemm_ptg_order_with_time(frame):
        return animate_gemm_common(frame, "ptg", "c", "tasks")
    
    def animate_gemm_ptg_order_with_time_all(frame):
        return animate_gemm_common(frame, "ptg", "abc", "tasks")
    
    def animate_gemm_ptg_order_core_swaps(frame):
        return animate_gemm_common(frame, "ptg", "c", "swaps")
    
    def animate_gemm_ptg_order_progress(frame):
        return animate_gemm_common(frame, "ptg", "c", "progress")
        
    def animate_gemm_ptg_order_progress_all(frame):
        return animate_gemm_common(frame, "ptg", "abc", "progress")
    
    def animate_potrf_common(frame, mode):
        global C_status
        global tasks_at_frame
        global C_expected
        global large_tile
        # time_point_curr and time_point_prev in units of nanoseconds
        time_point_curr = ((frame+0)*(last_end - first_begin))//num_frames + first_begin
        time_point_prev = ((frame-1)*(last_end - first_begin))//num_frames + first_begin
        if(frame == 0): # One empty frame at start
            time_point_prev = time_point_cur = first_begin
        if(frame >  num_frames): # This construction to allow dead frames at the end
            time_point_prev = time_point_curr = last_end
            
        if(mode == "tasks"):
            C_status = C_status * 0 # Always zero when doing task timing
            
        tasks_before = orderdf.loc[orderdf["end"] <= time_point_curr]
        tasks_during = tasks_before.loc[(tasks_before["end"] > time_point_prev)]
            # aside: you might think this construction with > time_point_prev wouldn't include the first task,
            #        but this is fine! The only assumption made is that the first tasks does not end the same nanoseond
            #        that it begins. That sounds like a safe assumption to me.
        
        tasks_at_frame.append(len(tasks_during))
        
        for index, task in tasks_during.iterrows():
            m = task["m"]
            n = task["n"]
            k = task["k"]
            # print("type:", task["type"], "m,n,k:", m,n,k)
            # uplo of the matrix order the m, n, and k differently
            if running_system == "dplasma": # Upper triangular
                if task["type"] == name_to_task_num["potrf"]:
                    if k is None:
                        print("k should not be none here")
                    target_row = k
                    target_col = k
                elif task["type"] == name_to_task_num["trsm"]:
                    if potrf_uplo == "upper":
                        target_row = k
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = k
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (trsm)")
                elif task["type"] == name_to_task_num["syrk"]:
                    if potrf_uplo == "upper":
                        target_row = n
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = m
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (syrk)")
                elif task["type"] == name_to_task_num["gemm"]:
                    if m is None:
                        print("m should not be none here")
                    if n is None:
                        print("n should not be none here")
                    target_row = m
                    target_col = n
                else:
                    print("error: unexpected task of type %d" % task["type"])
                    print(task)
                C_status[target_row, target_col] += 1 / C_expected[target_row, target_col]
            elif running_system == "hicma": # Lower triangular
                stride = bigtilesize // tilesize
                if task["type"] == name_to_task_num["large-potrf"]:
                    if k is None:
                        print("k should not be none here (potrf)")
                    target_row = k
                    target_col = k
                    large_task = True
                    large_tile = k
                elif task["type"] == name_to_task_num["large-trsm"]:
                    if potrf_uplo == "upper":
                        target_row = k
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = k
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (trsm)")
                    large_task = True
                elif task["type"] == name_to_task_num["large-syrk"]:
                    if potrf_uplo == "upper":
                        target_row = n
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = m
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (syrk)")
                    large_task = True
                elif task["type"] == name_to_task_num["large-gemm"]:
                    if m is None:
                        print("m should not be none here (gemm)")
                    if n is None:
                        print("n should not be none here (gemm)")
                    target_row = m
                    target_col = n
                    large_task = True
                elif task["type"] == name_to_task_num["small-potrf"]:
                    if k is None:
                        print("k should not be none here (potrf)")
                    target_row = k
                    target_col = k
                    large_task = False
                elif task["type"] == name_to_task_num["small-trsm"]:
                    if potrf_uplo == "upper":
                        target_row = k
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = k
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (trsm)")
                    large_task = False
                elif task["type"] == name_to_task_num["small-syrk"]:
                    if potrf_uplo == "upper":
                        target_row = n
                        target_col = n
                    elif potrf_uplo == "lower":
                        target_row = m
                        target_col = m
                    if target_row is None or target_col is None:
                        print("a parameter was none that shouldn't have been (syrk)")
                    large_task = False
                elif task["type"] == name_to_task_num["small-gemm"]:
                    if m is None:
                        print("m should not be none here (gemm)")
                    if n is None:
                        print("n should not be none here (gemm)")
                    target_row = m
                    target_col = n
                    large_task = False
                else:
                    print("error: unexpected task of type %d" % task["type"])
                    print(task)
                if large_task == True:
                    for i in range(target_row*stride, (target_row+1)*stride):
                        for j in range(target_col*stride, (target_col+1)*stride):
                            if(task["type"] == name_to_task_num["large-potrf"]):
                                # This is a recursive task: no updates.
                                # The updates are spawned by its children
                                continue
                            elif(task["type"] == name_to_task_num["large-syrk"]):
                                if(j > i):
                                    continue # No suprise
                                C_status[i, j] += stride / C_expected[i, j]
                            elif(task["type"] == name_to_task_num["large-trsm"]):
                                C_status[i, j] += ((j % stride)+1) / C_expected[i, j]
                            elif(task["type"] == name_to_task_num["large-gemm"]):
                                C_status[i, j] += stride / C_expected[i, j]
                            else:
                                print("(small) task %d over-diag" % task["type"])
                else:
                    target_row += stride * large_tile
                    target_col += stride * large_tile
                    C_status[target_row, target_col] += 1 / C_expected[target_row, target_col]
            else:
                print("Error: unknown running system")
                
        axC.set_title("Matrix C at time t=%fs (%s)" % (time_point_curr/10**9, title))
        axC.pcolormesh(C_status, vmin = 0, vmax = 1)
        return
    
    def animate_potrf_tasks(frame):
        return animate_potrf_common(frame, "tasks")
        
    def animate_potrf_progress(frame):
        return animate_potrf_common(frame, "progress")
    
    if(task_type == "gemm"):
        if(trace_type == "dtd"):
            if(which_animate == "tasks"):
                animation_func = animate_gemm_dtd_order_with_time
                animation_init = animate_init_c
            elif(which_animate == "progress"):
                animation_func = animate_gemm_dtd_order_progress
                animation_init = animate_init_c
            elif(which_animate == "coreswaps"):
                animation_func = animate_gemm_dtd_order_core_swaps
                animation_init = animate_init_c
            elif(which_animate == "abctasks"):
                animation_func = animate_gemm_dtd_order_with_time_all
                animation_init = animate_init_abc
            elif(which_animate == "abcprogress"):
                animation_func = animate_gemm_dtd_order_progress_all
                animation_init = animate_init_abc
            else:
                print("error: no animation functions")
        elif(trace_type == "ptg"):
            if(which_animate == "tasks"):
                animation_func = animate_gemm_ptg_order_with_time
                animation_init = animate_init_c
            elif(which_animate == "progress"):
                animation_func = animate_gemm_ptg_order_progress
                animation_init = animate_init_c
            elif(which_animate == "coreswaps"):
                animation_func = animate_gemm_ptg_order_core_swaps
                animation_init = animate_init_c
            elif(which_animate == "abctasks"):
                animation_func = animate_gemm_ptg_order_with_time_all
                animation_init = animate_init_abc
            elif(which_animate == "abcprogress"):
                animation_func = animate_gemm_ptg_order_progress_all
                animation_init = animate_init_abc
            else:
                print("error: no animation functions")
        else:
            print("error: unrecognized trace type")
    elif (task_type == "potrf"):
        animation_init = animate_init_c
        if(which_animate == "tasks"):
            animation_func = animate_potrf_tasks
        elif(which_animate == "progress"):
            animation_func = animate_potrf_progress
        
    mid_time = time.time()
    # Supply a half second of stillness at the end
    padding_frames = max(fps // 2, 2)
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
    print("execution time to generate graphs: %f seconds" % (end_time-mid_time))
    print("execution time to preprocess data: %f seconds" % (mid_time-start_time))
    print("execution time total: %f seconds" % (end_time-start_time))
    
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
    ax.set_xlabel("task number (sorted by task duration)")
    ax.set_ylabel("task duration (ms)")
    fname = "tasks_times_sorted_(%s).png" % title
    plt.savefig(fname)
    # Offer the user the filename in case they wish to display it after
    print("saved task metadata file:", fname)
    
    if(sum(tasks_at_frame) != expected_tasks):
        print("error: expected %d tasks but I only observed %d in the trace" %
              (expected_tasks, sum(tasks_at_frame)))
    
    """ # This test would only make sense for 'progress' mode
    if(sum(sum(C_status)) != expected_tasks):
        print("warning: C_status expected updates %d times but received %d updates" %
              (expected_tasks, sum(sum(C_status))))
    """
    
    print()
