import matplotlib.pyplot as plt
import numpy as np

def combine_lsts_ignoring_last_elt(lst_0, lst_1, lst_2): #ignore last timer and set all three lists (without last entry) in one array.
    return np.array([lst_0[:-1], lst_1[:-1], lst_2[:-1]])

def data_plot(lst): #generate averaged data for pie chart
    total_av_lst = np.sum(lst[:,2])/ np.shape(lst)[0]
    lst[:,2] = lst[:,2] - (np.sum(lst,axis=1) - lst[:,2])  #calculate idle
    average_lst = np.sum(lst, axis=0)/ np.shape(lst)[0]     #calculate average for each funtion/idle
    return average_lst, total_av_lst

def pie_chart(lst, n, num_threads, title): #plot a pie chart
    labels_fnc = ['apply_stencil3d', 'axpby', 'idle', 'dot', 'init']            #labels for pie chart
    save_title = title + "_run_time_profile_" + str(n) + "_" + str(num_threads) #title for reading and saving
    textprops = {"fontsize":18}                                                 #font size of text in pie chart
    wedgeprops = {"linewidth": 1, "edgecolor":"white"}                          #width = 1
    plt.figure(figsize=[20, 10])                                                #figure size
    plt.pie(lst, labels=labels_fnc, autopct='%0.2f%%', radius = 1.2, textprops = textprops, wedgeprops = wedgeprops, startangle = 0)    #plot pie chart
    plt.title("Runtime profile of CG_solver\n", fontsize = 30)                     #title of pie chart
    plt.suptitle(f"\n\nfor n = {n} and OMP_NUM_THREADS = {num_threads}", fontsize = 20)   #subtitle of pie chart
    #plt.show()                                                                 #show figure
    plt.savefig(save_title + ".png")                                            #save figure

def bar_plot(lst0, lst1, n, num_threads, title):
    barWidth = 0.5
    br0 = np.arange(len(lst0))
    br1 = [x + barWidth for x in br0]
    labels_fnc = ['apply_stencil3d', 'axpby', 'idle', 'dot', 'init', 'cg_solver']#labels for bar plot
    save_title = title + "_bar_plot_" + str(n) + "_" + str(num_threads)         #title for reading and saving
    textprops = {"fontsize":18}                                                 #font size of text in bar plot
    wedgeprops = {"linewidth": 1, "edgecolor":"white"}                          #width = 1
    plt.figure(figsize =(24, 15))                                                #figure size
    bar0 = plt.bar(br0, lst0, color = 'r', width = barWidth, edgecolor ='k', label = '-O2 -g')                                            #plot bar based on lst0
    bar1= plt.bar(br1, lst1, color = 'b', width = barWidth, edgecolor ='k', label = '-O3 -march=native')                                            #plot bar based on lst1
    plt.bar_label(bar0, np.round(lst0,4))
    plt.bar_label(bar1, np.round(lst1,4))
    plt.xticks([-barWidth/2 + x + barWidth for x in range(len(lst0))], labels_fnc, fontsize=16)
    #plt.yticks(np.hstack((lst0,lst1)))
    plt.title("Bar plot of CG_solver timers\n\n", fontsize = 30)                  #title of bar plot
    plt.suptitle(f"\n\nfor n = {n} and OMP_NUM_THREADS = {num_threads}", fontsize = 20)   #subtitle of bar plot
    plt.legend()
    plt.xlabel("Basis functions and CG_solver", fontsize = 19)
    plt.ylabel("Execution time (s)", fontsize = 19)
    #plt.show()                                                                 #show figure
    plt.savefig(save_title + ".png")                                            #save figure
    
if __name__ == "__main__":
    #--------------Data----------------------------------------------------------------------------------------------------------------------
    n = 600
    num_threads = 12
    
    #lst = ['apply_stencil3d', 'axpby', 'idle', 'cg_solver', 'dot', 'init', 'main_cg_solver']
    #-O2 -g compiler
    lst_0 = [48.5, 11.73, 65.54, 4.952, 0.08326, 66.06]
    lst_1 = [48.55, 11.9, 65.91, 5.103, 0.08896, 66.44]
    lst_2 = [48.46, 11.64, 65.55, 5.091, 0.08854, 66.07]
    lst = combine_lsts_ignoring_last_elt(lst_0, lst_1, lst_2)                   #ignore last timer and set all runs in one array.

    #-O3 -march=native compiler
    agg_lst_0 = [38.03, 11.12, 54.38, 4.878, 0.0811, 54.86]
    agg_lst_1 = [38.34, 12.7, 56.68, 5.291, 0.0894, 57.16]
    agg_lst_2 = [38.11, 11.82, 55.3, 5.007, 0.08896, 55.78]
    agg_lst = combine_lsts_ignoring_last_elt(agg_lst_0, agg_lst_1, agg_lst_2)
    #-------------Averaged-data---------------------------------------------------------------------------------------------------------------
    average_lst, total_av_lst = data_plot(lst)                                                #calculate the average value of the lst wrt to each function
    agg_average_lst, agg_total_av_lst = data_plot(agg_lst)                                        #calculate the average value of the agg_lst wrt to each function
    #------------Plot-pie-chart---------------------------------------------------------------------------------------------------------------
    pie_chart(average_lst, n, num_threads, "o2")                                #plot pie chart as runtime profile based on averaged_lst
    pie_chart(agg_average_lst, n, num_threads, "o3")                            #plot pie chart as runtime profile based on agg_averaged_lst
    #------------Plot-bar---------------------------------------------------------------------------------------------------------------    
    bar_plot(np.hstack((average_lst, total_av_lst)), np.hstack((agg_average_lst, agg_total_av_lst)), n, num_threads, "two")
