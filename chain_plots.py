import numpy as np
import matplotlib.pyplot as plt

def replace( r, i, v) :
    ro = r
    r[i] = v
    return ro

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_prior_post( lnprior, chain, i, rngx ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    nwalk, nstep, nt = np.shape(chain)    
    x = np.linspace(rngx[0],rngx[1],101)
    fig, ax1 = plt.subplots()

    theta = np.median( chain[:,-1,:],axis=0)
    ax1.plot(x, list(map( lambda xx : np.exp(lnprior(replace(theta,i,xx))) , x )) )
    ax2 = ax1.twinx()
    ax2.hist( chain[:,:,i].flatten(), bins=np.linspace(rngx[0],rngx[1],101) )
    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_prob( prob, fname=None, color='red' ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    nwalk, nstep = np.shape(prob)
    steps = range(nstep)
    if( nwalk < 50 ):
        for i in range(nwalk):
            plt.plot( steps, prob[i,:], color="k", alpha=0.10)
    else:
        prob_p = np.zeros([5,nstep])
        for i in steps:
            prob_p[:,i] = np.percentile( prob[:,i],[2.5,25,50,75,97.5] )
        plt.fill_between( steps, prob_p[0], prob_p[4], color=color,alpha=0.1)
        plt.fill_between( steps, prob_p[1], prob_p[3], color=color,alpha=0.4)
        plt.plot(         steps, prob_p[2],            color=color,lw=2 )
    if fname:
        plt.savefig(fname)
        plt.close()
    #else:
        #plt.show()    
    return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_post_p( ax, chain, func, x, nburn=50, nsamp=100, **kwargs ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   ''' plots samples theta out of the a chain through the func( theta, x )'''
   nwalk, nstep, ntheta = np.shape(chain)
   nx = len(x)

   for i in range(nsamp):
      i_w = np.random.randint(     0, nwalk )
      i_s = np.random.randint( nburn, nstep )
      theta_t = chain[ i_w, i_s ]
      ax.plot( x, func(theta_t,x), **kwargs )
   return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_post( func, chain, x,y, nsamp=100 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    nwalk, nstep, ntheta = np.shape(chain)
    xu = np.sort(np.unique(x))
    xp = np.linspace(0,10,1001)
    for i in range(nsamp):
    	 nw = np.random.randint( 0,nwalk)
    	 ns = np.random.randint(50,nstep)
    	 theta = chain[nw,ns]
    	 mdata = func( xp,theta )
    	 plt.plot(  xp,     mdata, color="k", alpha=0.10)

    for xx in xu:
        plt.plot( xx, np.median( y[np.abs(x-xx)<0.05]), 'o', color='red')
    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_var( nv, chain, labels=None, fname=None  ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    nwalk, nstep, nvar = np.shape(chain)
    steps = range(nstep)
	
    f, axs = plt.subplots(3,3, sharex=True)

    for j,ax in enumerate(axs.reshape(-1)):
        if  nv+j<nvar  :
            for i in range(nwalk):
                if( labels is not None ):
                    ax.set_title(labels[nv+j])
                ax.plot( steps, chain[i,:,nv+j], color="k", alpha=0.10)
    if fname:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return
