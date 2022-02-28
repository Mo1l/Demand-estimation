
from cmath import inf
import numpy as np
from numpy.random import uniform
from numpy.linalg import inv, det
from tqdm.notebook import tqdm



# What I need: 
# A function that produces the Demand
# A function that produces supply. 


class endog_data(): 
    
    def cov_term(self, f, j, T, cov = None):
        if cov is None:
            sigma, omega = 0, 0 
        else: 
            unob_traits=np.random.multivariate_normal([0,0],cov, size = (f*j*T))
            #sigma, omega = np.maximum(0.0,unob_traits[:,0]).reshape(f*j,T), np.maximum(0.0,unob_traits[:,1]).reshape(f*j,T)  #Censoring at zero so marginal costs do not end up being negative
            sigma, omega = unob_traits[:,0].reshape(f*j,T), unob_traits[:,1].reshape(f*j,T)
    
        return sigma, omega 

    # Solving for endogeneous prices:
    def Consumer_demand(self, theta, X, P, sigma): 
        """ Returns market shares given product traits and "mean" parameters and delta MS / delta P_j + I'm adding an outside good. 

        Keyword arguments:
        P -- is a f, j dimension vector.
        mc -- marginal costs
        S -- demand scheme (a set of product-specific traits should be given as input). 
        """


        alpha = theta[0]
        beta = theta[1:]



        Pa = P * alpha # J x 1 on a 1 x 1 
        Xb = X @ beta # J by k on a k by 1 = J by 1 

        V = Pa + Xb + sigma

        # adding an outside option: 
        V = np.append(0,V)

        V_max = np.max(V)  
        V0 = V - V_max

        S = np.exp(V0) / np.sum(np.exp(V0))
        #S = np.clip(S, 0.0001,0.999)

        # Removing the Outside option again:
        S=S[1:] 
        S_v = S.reshape(-1,1) # Jx1
        # deriving own-price and cross-price derivatives 
        J = S.shape[0]
        
        #print(S_v.shape)
        Lamda = np.diag(alpha * S)
        SS_T = S_v @ S_v.T
        
        Gamma = (np.identity(J) - 1) * alpha * SS_T
        
        dSdp = Lamda + Gamma    # -alpha * S**2

        
        return S, dSdp, Lamda, Gamma


    def Oligo_prices(self, f:int,mc: np.array, S, dSdp, product_map):
        """ Returns a set of prices given a supply scheme and marginal costs.

        Keyword arguments:
        f -- number of companies
        mc -- marginal costs
        S -- demand scheme (a set of product-specific traits should be given as input). 
        """

        J = S.shape[0]

        Delta = -dSdp

        check=np.arange(J)
        # This loop is for setting derivatives equal to 
        for f, f_prods in product_map.items(): 
            not_f_prods=[j for j in check if j not in f_prods ]
            for i in f_prods:
                Delta[i,not_f_prods] = 0

        P = mc + inv(Delta) @ S # (mc includes omega) - I think this is 

        return P
    
    def hadamard(self, J, product_map):
        # Constructs a production mapping: used to setting derivatives equal to zero. 
        check=np.arange(J)
        hadamard = np.ones((J,J))
        for f, f_prods in product_map.items(): 
            not_f_prods=[j for j in check if j not in f_prods ]
            for i in f_prods:
                hadamard[i,not_f_prods] = 0
        return hadamard    
    
    def decomposition_prices(self, f:int, mc: np.array, P:np.array, S, Lamda, Gamma, omega, product_map, H):
        """ Returns a set of prices given a supply scheme and marginal costs. The Morrow and Skerlos (2010) "contraction" mapping is used. 

        Keyword arguments:
        f -- number of companies
        mc -- marginal costs
        S -- demand scheme (a set of product-specific traits should be given as input). 
        """
        
        Gamma_H =   H * Gamma
        Lamda_inv = inv(Lamda)



        Epsilon_func = Lamda_inv @ Gamma_H @(P - mc) - Lamda_inv @ S 
        
        P = mc + Epsilon_func + omega 

        return P


    def c_map(self, P, theta, X, mc, product_map, H, f, sigma, omega):
        # First step: Calculate market shares:
        S, dSdp, Lamda, Gamma = self.Consumer_demand(theta, X, P, sigma) 
        # Second step: Calculate new prices:
        #try:
        P = self.decomposition_prices(f, mc, P, S, Lamda, Gamma, omega, product_map, H)
        if (P < 0).any():
            print("P is smaller than 0, drawing new starting values" , end="\r")
            P = uniform(0.1,10, size = P.shape)
        #except np.linalg.LinAlgError:
            # If any singular matrix error occur then try with different starting values and pray. 
            #print("Singular matrix - trying to recalibrate ...")
            #P = uniform(0.1,10, size = P.shape)
            #P = self.c_map(P, theta, X, mc, product_map, H, f, sigma)

        #P = P.reshape((f,-1))
        return P

    # Simulating data: 

    def simulate_product_traits(self, f:int, j:int, k:int, which_type: str):
        """ Returns product traits and production info

        Keyword arguments:
        f -- number of companies
        j -- number of products. if Identical 2 leads to four products as there are two firms producing the same product. 
        k -- number of product traits. 
        which_type -- default is Identical meaning producers produce identical products. 
        """


        if which_type == "Identical":
            traits_0 = uniform(low=0, high=1, size=(1,j,k))

            product_map = dict()
            traits= traits_0.copy()
            for i in range(f): 
                product_map[i] = np.arange(i*j,(i+1)*j)
                if i!=0:
                    traits = np.append(traits,traits_0, axis=0)


        if which_type == "Differentiated":
            traits_0 = uniform(low=0, high=1, size=(f,j,k))

            product_map = dict()
            traits= traits_0.copy()
            for i in range(f): 
                product_map[i] = np.arange(i*j,(i+1)*j)

        return traits, product_map


    def sim_data(self, P, theta, X, mc, product_map, H, f, J, t=0):
        
        c_map_l = lambda P: self.c_map(P, theta, X, mc, product_map, H, f, self.sigma[:,t], self.omega[:,t])


        # record keeping:
        eps = np.finfo(float).eps    
        i=0

        # Initialize P0: 
        P0 = c_map_l(P)

        while np.all(np.abs(P0 - P) > eps) :
            P = c_map_l(P0)

            i =+ 1
            if i == 1000:
                print("Did Not Converge!") 
                break

            # Initialize new P0
            P0= P

        # wished values: 

        S, dSdp, Lamda, Gamma = self.Consumer_demand(theta, X, P, self.sigma[:,t]) 
        
        inv_true = (det(dSdp) != 0) 


        return P, S, inv_true

    # "Dynamics" 
    

    def sim_data_T(self, X, f, j, T, theta, gamma, cov, product_map, H, disable = False):
        
        # Draw some correlated noise: 
        self.sigma, self.omega =self.cov_term(f,j, T, cov)
        
        # simulate MC: 
        mc=self.mc_sim(X, gamma, T)
        
        # Determine eq prices and 
        P = np.zeros((j*f,T)) + np.nan
        S = np.zeros((j*f,T)) + np.nan
        #cov = np.array([[1,0.2],[0.2,1]])

        P0 = np.zeros((f*j)) +1               #Initital price vector

        for t in tqdm(range(T), disable = disable): 
            inv_true = False
            while not inv_true: 
                P[:,t], S[:,t], inv_true=self.sim_data(P0, theta, X, mc, product_map, H, f, j, t)
                P0 = uniform(0.1,10, size = (f*j))

        return P, S 


    def mc_sim(self, X, gamma, T=1, noise_type="Triangular", dynamics=None, use_X = "Yes"):
        """ Returns a set of marginal costs for each product (potentially) based on traits.

        Keyword arguments:
        X -- product traits
        gamma -- cost factors on X 
        T -- Number of periods (can be used to extend to multiple markets. 
        noise_type -- which noise type to use 
        dynamics -- whether there should be any shocks to mc over time. (not implemented)
        use_X -- if "Yes" then marginal costs depends on traits. Otherwise x = ones.  
        """

        J = X.shape[0]

        if noise_type == "Triangular":
            eps = np.random.triangular(left=0, mode=0.5, right=1, size=(J,T)).reshape(J,T)

        # I will leave the 
        if use_X != "Yes": 
            mc = gamma + eps 
        else: 
            mc = (X@gamma).reshape(J) #+ self.omega # + eps #+ np.broadcast_to(self.omega.reshape(-1,1), (J,T))
            #J

        return mc


    
