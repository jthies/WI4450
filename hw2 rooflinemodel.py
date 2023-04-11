import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def ci_gflops(intensity, byte): #list, matrix
    flops = np.zeros((np.shape(byte)[0], np.shape(byte)[1],len(intensity)))
    for i in range(np.shape(byte)[0]):
        for j in range(len(intensity)):
            flops[i,:,j] = intensity[j] * byte[i,:]
    return flops

def rate(n_lst,t_lst):
    #r = np.zeros(np.shape(n_lst))
    r = np.zeros(np.shape(t_lst))
    #it = [16, 64, 256, 384, 1024, 1280, 1536]
    for i in range(np.shape(t_lst)[0]):
        r[i,:] = n_lst / t_lst[i,:] #n_lst[i,:] / t_lst[i]
    return r

def computational_intesisty_matrix(flop, byte):
    m = np.shape(byte)[0]
    n = np.shape(byte)[1]
    ci = np.zeros((m,n))
    for i in range(np.shape(byte)[0]):
        ci[i,:] = flop[i,:] / byte[i,:]
    return ci

def computational_intesisty_vector(flop, byte):
    return flop / byte
        

def plot(ci_lst, GBytesRate, likwid_GBytesRate, num_threads, title): #plot
    flop_lst = ci_gflops(ci_lst, GBytesRate)
    likwid_flop_lst = ci_gflops(ci_lst, likwid_GBytesRate)
    
    subplots_titles = ['apply_stencil3d', 'axpby', 'dot', 'init']#, ['Likwid: apply_stencil3d', 'Likwid: axpby', 'Likwid: dot', 'Likwid: init']]
    save_title = title + str(600**3) + "_" + str(num_threads)          #title for reading and saving
    m = np.shape(flop_lst)[0]
    n = np.shape(flop_lst)[1]-1
    fig, ax = plt.subplots(m,n,figsize=[20, 10])                        #figure size
    
    for i in range(m):#np.shape(flop_lst)[0]):
        ax[i,0].set_ylabel("GFlops/s")
        Rmax = (num_threads[i]/48) * np.array([32.75, 21.83333246, 32.75, 0])
        for j in range(n):
            max_flop_lst = np.minimum(262*ci_lst, Rmax[j])
            flp_lst = np.minimum(flop_lst[i,j,:], Rmax[j])
            likwid_flp_lst = np.minimum(likwid_flop_lst[i,j,:], Rmax[j])
            ax[i,j].plot(ci_lst, max_flop_lst, 'b', label = "using max Bandwidth")
            ax[i,j].plot(ci_lst, likwid_flp_lst, 'r', linestyle='dashed', label = "Likwid-bench")
            ax[i,j].plot(ci_lst, flp_lst, 'lightgreen', linestyle='dotted', label = "Benchmark")                                       
            ax[i,j].set_title(subplots_titles[j] + f': num_threads = {num_threads[i]}', fontsize = 6)
            ax[i,j].set_xlim([ci_lst[0],ci_lst[-1]])
            
            ax[i,j].annotate(str(round(Rmax[j],2)) + "Gflp/s",xy=(ci_lst[-5],max_flop_lst[-5]+1), color='black')
            ax[i,j].annotate(str(262) + "GB/s",xy=(ci_lst[0],2**(-5)), color='darkblue')
            ax[i,j].annotate(str(round(likwid_GBytesRate[i,j],2)) + "GB/s",xy=(ci_lst[0],2**(-10)),color='darkred')
            ax[i,j].annotate(str(round(GBytesRate[i,j],2)) + "GB/s",xy=(ci_lst[0],2**(-14)), color='darkgreen')

            ax[i,j].set_ylim([2**(-15),2**7])
            ax[i,j].set_xscale('log', base=2)
            ax[i,j].set_yscale('log', base=2)
            if i < m-1:
                ax[i,j].xaxis.set_visible(False)
                ax[i,j].set_xlabel("Gflops/GBytes")
            else:
                ax[i,j].set_xlabel("GFlops/GBytes")
                
            
            
    fig.suptitle("Roofmodel of operations \n", fontsize = 30)       #title
    plt.legend(loc='lower right',bbox_to_anchor=(1.45, -1.25))
    plt.show()                                                      #show figure
    fig.savefig(save_title + ".png")   

def plot_bandwidth(num_threads, GBytesRate, likwid_GBytesRate, title): #plot    
    subplots_titles = ['apply_stencil3d', 'axpby', 'dot', 'init']#, ['Likwid: apply_stencil3d', 'Likwid: axpby', 'Likwid: dot', 'Likwid: init']]
    save_title = title + str(600**3) + "_" + str(num_threads[-1])          #title for reading and saving
    m = np.shape(GBytesRate)[1] // 2
    n = m
    fig, ax = plt.subplots(m,n,figsize=[20, 10])                        #figure size
    k = 0
    for i in range(m):
        for j in range(n):
            ax[i,j].plot(num_threads, likwid_GBytesRate[:,k], 'r', linestyle='dashed', label = "Likwid-bench")
            ax[i,j].plot(num_threads, GBytesRate[:,k], 'darkgreen', linestyle='dotted', label = "Benchmark")                                       
            ax[i,j].set_title("\n\n\n" + subplots_titles[k], fontsize = 16)
            ax[i,j].set_xlim([num_threads[0],num_threads[-1]])
            ax[m-1,j].set_xlabel("OMP_NUM_THREADS\n", fontsize=14)
            ax[i,0].set_ylabel("GByte/s", fontsize=14)
            k+=1
            
    fig.suptitle("Bandwidth of operations \n", fontsize = 30)       #title
    plt.legend(loc='lower right')
    plt.show()                                                      #show figure
    fig.savefig(save_title + ".png")

def plot_roofmodel(ci_lst, likwid_GBytesRate, max_Gflops, thread_number,ylim):
    ci = [0.81124998, 2**(-3), 2**(-3), 0]
    subplots_titles = ['apply_stencil3d', 'axpby', 'dot', 'init']
    save_title = "Roofmodel_" + str(600**3) + "_" + str(thread_number)
    m = np.shape(GBytesRate)[1]//2
    n = m
    fig, ax = plt.subplots(m,n,figsize=[20, 10])
    k = 0
    if thread_number == 12:
        row = 6
    elif thread_number == 48:
        row = 15
    for i in range(m):
        for j in range(n):
            likwid_GflopsRate = np.minimum(likwid_GBytesRate[row,k]*ci_lst, max_Gflops[k])
            Rmax = ci[k]*likwid_GBytesRate[row,k]
            ax[i,j].plot(ci_lst, likwid_GflopsRate)
            ax[i,j].plot(np.linspace(ci_lst[0],ci[k],13), Rmax*np.ones(13), 'r--')
            ax[i,j].plot([ci[k],ci[k]], [0,Rmax], 'r--')
            ax[i,j].set_title("\n\n\n" + subplots_titles[k], fontsize = 16)
            ax[i,j].set_ylabel("GFlops/s", fontsize=14)
            ax[i,j].set_xscale('log', base=2)
            ax[i,j].set_yscale('log', base=2)
            ax[i,j].set_xlim([ci_lst[0], ci_lst[-1]])
            ax[i,j].set_ylim(ylim)
            ax[i,j].annotate(str(round(max_Gflops[k],3)) + "GFlop/s",xy=(ci_lst[-6],max_Gflops[k]+1), color='black')
            ax[i,j].annotate(str(round(likwid_GBytesRate[row,k],3)) + "GBytes/s",xy=(ci_lst[1],2**(-2)), color='darkblue')
            if k<3:
                ax[i,j].annotate(str(round(Rmax,3)) + "GFlop/s",xy=(ci_lst[7],Rmax+1), color='r')
            else:
                ax[i,j].annotate(str(round(Rmax,3)) + "GFlop/s",xy=(ci_lst[0],Rmax+2**(-11)), color='r')
            if i < m-1:
                ax[i,j].xaxis.set_visible(False)
            else:
                ax[i,j].set_xlabel("flops/bytes", fontsize=14)
            k += 1

    fig.suptitle("Roofmodel for " + str(thread_number) + " OMP_NUM_THREADS \n", fontsize = 30)
    plt.show()                                                      #show figure
    fig.savefig(save_title + ".png")
if __name__ == "__main__":
    #--------------Data----------------------------------------------------------------------------------------------------------------------
    n_x = 600
    n_y = 600
    n_z = 600
    num_threads1 = [1,2,4,6,8,10,12]
    num_threads2 = [16,20,24,28,32,36,40,44,48]
    num_threads = np.hstack((num_threads1, num_threads2))
    m = n_x**3
    #-------- -O3 -march=native compiler------------------
    print("[apply_stencil3d, axpby, dot, init]")
    convert = 10**(-9)
    f_apply = (6+7)*(n_x-2)*(n_y-2)*(n_z-2) + (5+6)*2*((n_x-2)*(n_y-2)+(n_x-2)*(n_z-2)+(n_z-2)*(n_y-2)) + (4+5)*4*((n_x-2) + (n_y-2) + (n_z-2)) + (3+4)*8
    gflp = convert * np.array([f_apply  , 3*m, 2*m, 0])
    # best case apply
    gb = convert * np.array([1 * 72 + 2 * m * 8, 3*8*m, 2*m*8, 1*m*8])
    print("gflp = \n", gflp)
    print("gbyte =\n", gb)
    mean_time_1 = [4.12477, 0.248209, 0.265777, 0.189335]
    mean_time_2 = [2.07987, 0.154129, 0.141776, 0.117934]
    mean_time_4 = [1.09873, 0.087185, 0.077678, 0.0669841]
    mean_time_6 = [0.753021, 0.0674098, 0.0508102, 0.0512074]
    mean_time_8 = [0.572153, 0.0764793, 0.0491571, 0.0466279]
    mean_time_10 = [0.462079, 0.0591158, 0.0400862, 0.0362699]
    mean_time_12 = [0.384966, 0.0541445, 0.0356683, 0.0326477]
    mean_time_12_opt = [0.375921, 0.0684936, 0.034154, 0.051886]
    mean_time_16 = [0.292667, 0.0738296, 0.0356342, 0.0574878]
    mean_time_20 = [0.236696, 0.0635878, 0.0314832, 0.0493674]
    mean_time_24 = [0.205735, 0.0583521, 0.0296553, 0.0475471]
    mean_time_28 = [0.181431, 0.0497365, 0.0256697, 0.0400297]
    mean_time_32 = [0.157115, 0.044099, 0.023035, 0.0359521]
    mean_time_36 = [0.140838, 0.0385734, 0.0206, 0.030765]
    mean_time_40 = [0.124651, 0.0352632, 0.018379, 0.0287777]
    mean_time_44 = [0.11623, 0.0319701, 0.0168511, 0.0258742]
    mean_time_48 = [0.108398, 0.0294012, 0.0157922, 0.023951]
    mean_time_48_opt = [0.103033, 0.0298411, 0.015929, 0.0242323]
    mean_time1 = np.vstack((mean_time_1, mean_time_2, mean_time_4, mean_time_6, mean_time_8, mean_time_10, mean_time_12_opt))
    mean_time2 = np.vstack((mean_time_16, mean_time_20, mean_time_24, mean_time_28, mean_time_32, mean_time_36, mean_time_40, mean_time_44, mean_time_48_opt))
    mean_time = np.vstack((mean_time1,mean_time2))
    print(len(mean_time))
    print("time = \n", mean_time)
    GflopsRate1 = rate(gflp, mean_time1)
    GBytesRate1 = rate(gb, mean_time1)
    GflopsRate = rate(gflp, mean_time)
    GBytesRate = rate(gb, mean_time)

    #----------Likwidth benchmarks----------------------
    #apply_stencil \approx 2 loads
    ##num_threads=1
    GBytesRate_1 = [16.28689, 21.19408, 16.28689,10.18579]

    ##num_threads=2
    GBytesRate_2 = [16.23484, 20.94541, 16.23484, 9.40662]

    ##num_threads=4
    GBytesRate_4 = [35.7024,  35.97644,  35.7024, 16.43751]

    ##num_threads=6
    GBytesRate_6 = [41.70115, 45.33566, 41.70115, 21.17401]

    ##num_threads=8
    GBytesRate_8 = [50.79889, 49.50399, 50.79889, 23.35985]

    ##num_threads=10
    GBytesRate_10 = [57.85589, 51.99277, 57.85589, 24.44285]

    ##num_threads=12
    GBytesRate_12 = [59.57001, 53.16909, 59.57001, 25.39708]

    ##num_threads=16
    GBytesRate_16 = [76.65338, 72.74962, 76.65338, 31.35577]

    ##num_threads=20
    GBytesRate_20 = [62.13021, 63.22537, 62.13021, 26.41007]

    ##num_threads=24
    GBytesRate_24 = [66.63166, 67.95885, 66.63166, 28.65257]

    ##num_threads=28
    GBytesRate_28 = [73.51328, 73.82627, 73.51328, 30.02957]

    ##num_threads=32
    GBytesRate_32 = [74.50333, 72.19568, 74.50333, 30.88870]

    ##num_threads=36
    GBytesRate_36 = [63.67282, 67.79748, 63.67282, 28.83170]

    ##num_threads=40
    GBytesRate_40 = [71.39553, 70.81015, 71.39553, 30.10999]

    ##num_threads=44
    GBytesRate_44 = [71.56815, 70.97796, 71.56815, 30.95856]

    ##num_threads=48
    GBytesRate_48 = [75.18786, 73.17101, 75.18786, 31.38866]

    likwid_GBytesRate1 = 2*np.vstack((GBytesRate_1, GBytesRate_2, GBytesRate_4, GBytesRate_6, GBytesRate_8, GBytesRate_10, GBytesRate_12)) #GBytes / s
    likwid_GBytesRate2 = 2*np.vstack((GBytesRate_16, GBytesRate_20, GBytesRate_24, GBytesRate_28, GBytesRate_32, GBytesRate_36, GBytesRate_40, GBytesRate_44, GBytesRate_48))
    likwid_GBytesRate = np.vstack((likwid_GBytesRate1, likwid_GBytesRate2))
    
    ci = computational_intesisty_matrix(GflopsRate, GBytesRate)#np.array([2**i for i in range(-6,6)])
    likwid_ci = computational_intesisty_vector(gflp, gb)

    print("flp rate = \n", GflopsRate)
    print("gb rate = \n", GBytesRate)
    print("gb likwid rate = \n", likwid_GBytesRate)
    
    print("ci = \n", ci)
    print("likwid_ci = \n", likwid_ci)
    print()
    ci_lst = np.array([2**i for i in range(-15,2)])
    
    #title = "Roofmodel"
    #plot(ci_lst, GBytesRate, likwid_GBytesRate, num_threads, title)
    max_Gflops = np.array([2.5*2*(8+8), 2.5*(2*8 + 1*8), 2.5*2*(8+8), 2.5])
    #num_threads = 12
    plot_bandwidth(num_threads1, GBytesRate1, likwid_GBytesRate1, "Bandwidth")
    ci_lst = np.array([2**i for i in range(-15,10)])
    max_Gflops12 = 12*max_Gflops
    plot_roofmodel(ci_lst, likwid_GBytesRate1, max_Gflops12, 12, [2**(-12), 2**(14)])
    #num_threads = 48
    plot_bandwidth(num_threads, GBytesRate, likwid_GBytesRate, "Bandwidth")
    ci_lst = np.array([2**i for i in range(-15,10)])
    max_Gflops48 = 48*max_Gflops
    plot_roofmodel(ci_lst, likwid_GBytesRate, max_Gflops48, 48, [2**(-12), 2**(14)])
    #--------------------------------------------------------------------task-4-------------------------------------------------------------------
