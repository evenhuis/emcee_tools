import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def optimise_ll( log_prob, theta, *args,  nsub=50, nloop=20,\
                live_plot=False, plot=False, quiet=False ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def nll( theta, *args ):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        return -log_prob( theta, *args )

    simplex=None
    X=[0] ; Y=[-nll( theta, *args )]

    if( live_plot ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel("Log Probability")
        ax.set_xlabel("Iteration")
        plt.grid()

        graph = ax.plot(X,Y,'-o')[0]

        ax.set_xlim(-5,nloop*nsub+5)
        ax.autoscale(True,axis='y')

        plt.draw()
        plt.pause(0.01)

    for i in range(1,nloop+1):
        result = op.minimize(nll, theta,  args=(args), method="nelder-mead", \
                 options={'initial_simplex':simplex,'xtol': 1e-4, 'disp':False , 'maxiter':50})
        theta = result.x
        simplex = result.final_simplex[0]

        X.append(i*nsub)
        Y.append(-result.fun)
        if( live_plot ):
            graph.set_xdata(X)
            graph.set_ydata(Y)
       
            yhi=np.percentile(Y,90)
            mid=0.5*(yhi+np.min(Y))
            rng=0.5*(yhi-np.min(Y))
            ax.set_ylim( mid-1.2*rng,mid+1.2*rng)
            plt.draw()
            plt.pause(0.01)
        if( not quiet ) : print("{:3d} {:15.3f}".format(i*nsub,-result.fun))
        if( result.success ) : break
    
    if( plot ):
        
        plt.plot( X,Y,'-o')
        plt.ylabel("Log Probability")
        plt.xlabel("Iteration")
        plt.axhline(Y[-1],ls='-',color='red')
        plt.axhline(Y[-1]-2,ls='--',color='red')
        plt.text(0.1*X[0]+0.9*X[-1],0.25*Y[0]+0.75*(Y[-1]),"tolerance",color='red')
        plt.grid()
        plt.suptitle("Maximise Liklihood")
        plt.show()
    return theta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def setup_initial_ball(lnprob, theta, *args, nwalker=300, scale=None ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' Takes a seed parameter vector (theta) and displaces it.
    It uses the lnprob function with optional (x predictor,y observation) values
    to ensure there are log probabilty is finite
     scale is an array
    '''
    ndim = np.size(theta)
    p0 = np.zeros( [nwalker, ndim] )

    ll = lnprob(theta, *args)

    for i in range(nwalker):
        lp = -np.inf
        while not ( lp>-np.inf) :
            theta_t = np.array(theta)
            for j in range(ndim):
                theta_t[j] = theta[j]*(1 + 1e-3*(np.random.random()-0.5)) + (1e-4*(np.random.random()-0.5))
            lp = lnprob( theta_t, *args)
            if( np.isfinite(lp) ):
                p0[i] = theta_t
    return p0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def change_walkers( lnprob, p0, nwalker, *args ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' either compresses or expands the walker set'''
    [nold,ndim] = np.shape(p0)
    if( nwalker < nold ):
        ind = np.arange(0,nold,dtype=int)
        ind = np.random.permutation( ind )
        p1 = p0[ ind,:]
        return p1

    if( nold < nwalker ):
        p1 = np.zeros( [nwalker,ndim] )
        p1[:nold,:] = p0
        nt = nold
        while( nt < nwalker ):
            # select two points
            i_1 = np.random.randint( 0, nold )
            i_2 = np.random.randint( 0, nold )
            if( i_1 == i_2 ) : next

            # create a point at their midpoint
            ir = np.random.rand()
            w1 =   + ir
            w2 = 1 - ir
            theta_t = w1*p0[i_1,:] + w2*p0[i_2,:]
            lp = lnprob( theta_t, *args) 
            if( np.isfinite(lp) ):
                p1[nt,:] = theta_t
                nt=nt+1
        return p1

    return p0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def setup_live_plot( nsteps, div,max_LL=None ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 

    fig = plt.figure()
    plt.ion()
    ax = fig.add_subplot(111)
    plt.grid()

    ax.set_xlim(-5,nsteps+5)
    ax.autoscale(True,axis='y')

    if( max_LL is not None ):
        ax.plot( [0,400],[max_LL,max_LL], '--', lw=1.4)
    plt.draw()
    plt.pause(0.01)
    return ax    # return empty X and Y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def update_live_plot( ax,result, X,percs,i,j,max_LL=None):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ax.clear()
    nsteps=len(X)
    ax.set_xlim(-5,nsteps+5)
    ax.grid()
    plt.fill_between(X[0:j+1],percs[0,0:j+1],percs[4,0:j+1], color='red',alpha=0.1)
    plt.fill_between(X[0:j+1],percs[1,0:j+1],percs[3,0:j+1], color='red',alpha=0.4)
    plt.plot(        X[0:j+1],percs[2,0:j+1],                color='red', lw=2 )
    if( max_LL is not None ):
        ax.plot( [0,400],[max_LL,max_LL], '--', lw=1.4,color="blue")
    ax.set_ylabel("log prob")
    ax.set_xlabel("iteration")
    plt.draw()
    plt.pause(0.01)
    return 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def MCMC_all( lnprob,theta, *args, nsteps=300, nwalker=200, threads=1, \
              max_LL=None, live_plot=False, quiet=False ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if( not quiet ):
        print("- - - - Starting MCMC sampling - - - - ")


    div=max(nsteps//100,1)
    nprint=nsteps//div
    X=np.zeros(nprint)
    percs=np.zeros([5,nprint])

    if( live_plot ):
        ax = setup_live_plot( nsteps,div, max_LL )

    div=1
    ndim, nwalkers = np.size(theta), nwalker
    ll = lnprob(theta, *args)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args), threads=threads)
    p0 = setup_initial_ball( lnprob, theta, *args, nwalker=nwalker )

    j=0
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        if( not quiet ):
            if (i+0) % 10 == 0:
                print("{0:5.1%} {1:}".format(float(i) / nsteps, np.average(result[1])))
        if (i+0) % div == 0:
            X[j]=i
            percs[:,j] = np.percentile( result[1], [5,25,50,75,95] )
            if(live_plot): update_live_plot(ax, result,X,percs,i,j,max_LL)
            j+=1
    
    plt.fill_between(X,percs[0],percs[4], color='red',alpha=0.1)
    plt.fill_between(X,percs[1],percs[3], color='red',alpha=0.4)
    plt.plot(        X,percs[2],          color='red', lw=2 )
    if( max_LL is not None ):
        plt.plot( [0,400],[max_LL,max_LL], '--', lw=1.4,color="blue")
    plt.ylabel("log prob")
    plt.xlabel("iteration")          
    plt.show()
    return sampler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def MCMC_restart( lnprob, p0, *args, nsteps=400, threads=1, \
                  plot=False, live_plot=False, max_LL=None ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
    print("- - - - Starting MCMC sampling - - - - ")
    nwalkers, ndim = np.shape(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args), threads=threads)

    div =1
    if( live_plot ): 
        div=max(nsteps//100,1)
        ax,X,percs = setup_live_plot( nsteps,div, max_LL )

    j=0
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):

        if (i+0) % div == 0:
            print("{0:5.1%} {1}".format(float(i) / nsteps, np.average(result[1])))
            if(live_plot):  X,percs = update_live_plot(ax, result,X,percs,i,j,max_LL)
            j+=1

    if( live_plot ):
        plt.ioff()
    return sampler


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def median_chain( chain, nburn=50 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    nwalk, nstep, ntheta = np.shape(chain)
    theta = np.zeros(ntheta)
    for i in range(len(theta)):
        theta[i] = np.median( chain[:,nburn:,i] )
    return theta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def pull_post( chain, func, nburn=50, nsamp=100  ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' evals func on trandom samples from cgain of thetaa, x )'''
    nwalk, nstep, ntheta = np.shape(chain)

    cout = np.zeros( nsamp )
    for i in range(nsamp):
        i_w = np.random.randint(     0, nwalk )
        i_s = np.random.randint( nburn, nstep )
        theta_t = chain[ i_w, i_s ]
        ya = func( theta_t )
        cout[i] = np.copy(ya)
    return cout



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def save_chain( filename, sampler ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    import pickle
    slist = [sampler.lnprobability, sampler.chain ]
    pickle.dump( slist, open( filename, "wb" ) )
    return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_chain( filename ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    import pickle
    return pickle.load( open( filename, "rb" ) )



