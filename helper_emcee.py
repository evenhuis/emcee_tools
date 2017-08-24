import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def optimise_ll( log_prob, theta, *args, live_plot=False, nsub=50, nloop=20 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def nll( theta, *args ):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        return -log_prob( theta, *args )

    simplex=None
    X=[0] ; Y=[nll( theta, *args )]

    if( live_plot ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        Y.append(result.fun)
        if( live_plot ):
            graph.set_xdata(X)
            graph.set_ydata(Y)
       
            yhi=np.percentile(Y,90)
            mid=0.5*(yhi+np.min(Y))
            rng=0.5*(yhi-np.min(Y))
            ax.set_ylim( mid-1.2*rng,mid+1.2*rng)
            plt.draw()
            plt.pause(0.01)
        print("{:3d} {:15.3f}".format(i*nsub,result.fun))
        if( result.success ) : break
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
                theta_t[j] = theta[j]*(1 + 1e-2*(np.random.random()-0.5)) + (1e-3*(np.random.random()-0.5))
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
def MCMC_all( lnprob,theta, *args, nsteps=300, nwalker=200 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("- - - - Starting MCMC sampling - - - - ")
    print(theta )
    ndim, nwalkers = np.size(theta), nwalker
    ll = lnprob(theta, *args)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args), threads=7)
    p0 = setup_initial_ball( lnprob, theta, *args, nwalker=nwalker )

    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):

        if (i+0) % 10 == 0:
            print("{0:5.1%} {1}".format(float(i) / nsteps, np.average(result[1])))

    return sampler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def MCMC_restart( lnprob, p0, *args, nsteps=400 ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
    print("- - - - Starting MCMC sampling - - - - ")
    nwalkers, ndim = np.shape(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args), threads=7)

    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):

        if (i+0) % 10 == 0:
            print("{0:5.1%} {1}".format(float(i) / nsteps, np.average(result[1])))
                   

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
def pull_post( chain, func, x, nburn=50, nsamp=100,  ):
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' evals func on trandom samples from cgain of thetaa, x )'''
    nwalk, nstep, ntheta = np.shape(chain)

    
    ta,ya = func( chain[0,0,:], x )
    [nv,nx] = np.shape(ya)
    cout = np.zeros( [nsamp,nv,nx] )

    for i in range(nsamp):
        i_w = np.random.randint(     0, nwalk )
        i_s = np.random.randint( nburn, nstep )
        theta_t = chain[ i_w, i_s ]
        ta , ya = func( theta_t, x )
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



