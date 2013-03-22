import kernels
import means
from gp import gp
from min_wrapper import min_wrapper
from solve_chol import solve_chol
import Tools.general
import Tools.nearPD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from utils import convert_to_array, hyperParameters, plotter, FITCplotter

if __name__ == '__main__':
    ## GENERATE data from a noisy GP
    n1 = 80; n2 = 40;
    S1 = np.eye(2); S2 = np.array([[1, 0.95], [0.95, 1]])
    m1 = np.array([0.75, 0]); m2 = np.array([-0.75, 0])

    x1 = np.array([[0.089450165731417,  -0.000700765006939],\
        [ 1.171605560541542,   1.177765337635947],\
        [ 1.404722675089394,  -0.017417915887421],\
        [ 0.556096196907929,  -1.489370243839215],\
        [ 1.213163445267992,   0.044545401368647],\
        [ 0.173404742510759,  -0.675668036759603],\
        [ 2.225008556585363,   0.469803193769368],\
        [ 1.470329290331445,   0.887642323697526],\
        [ 2.715199208821485,   0.621044646503113],\
        [ 0.173640760494328,  -0.936054178730056],\
        [ 2.038152815025167,   0.262587298316711],\
        [ 1.670218375320427,  -2.633186886994263],\
        [ 0.270098501389591,  -0.948779657473203],\
        [ 1.396339236138275,  -1.114992287201776],\
        [-1.482070589718501,  -0.654590652482805],\
        [-1.493788226272929,   0.382017940248275],\
        [ 1.025083846875763,  -0.860344923788873],\
        [ 0.750316336734172,  -0.101864205602753],\
        [ 0.184311310148912,  -0.258523866245887],\
        [ 0.221868667121623,  -1.393954437105630],\
        [ 2.258881477897777,  -0.786806071526136],\
        [ 1.211362530151533,  -0.423431246029886],\
        [ 1.525307406741207,  -0.097975367602030],\
        [ 0.978930232706465,   0.476154349549524],\
        [ 1.347884229346280,  -0.248408186838667],\
        [ 1.205779546204216,  -0.090878327349907],\
        [ 0.124388644862000,   0.599612645000285],\
        [ 0.784044356662233,   0.356596736271853],\
        [ 1.060216683845210,  -0.318474838087900],\
        [ 1.678114484474938,   0.678735373910422],\
        [ 0.973851135005570,   0.024880700382574],\
        [ 0.016237746864886,  -0.480899874254564],\
        [ 0.979406721923196,   0.697708815321128],\
        [ 2.217307638531248,  -0.956931847027775],\
        [ 2.150475558834153,   1.059031573329512],\
        [ 1.050502393215048,   0.532141747419667],\
        [ 1.210593098269218,  -0.318123542280113],\
        [ 0.426309208807901,  -0.571727978045793],\
        [ 0.742552105732714,  -0.122112766396886],\
        [ 0.757210723588679,   0.862002000781123],\
        [-0.431639130160791,  -0.763118261936640],\
        [-0.748398486307095,  -0.603667649379360],\
        [ 0.975086541108249,  -1.525297946453790],\
        [ 0.074503762788667,  -0.092155036190678],\
        [-0.668889572018935,   1.305400680048752],\
        [ 0.725632503186580,   0.096286255882168],\
        [-1.042270707136463,   1.297009698531055],\
        [ 1.943144890398260,  -1.051176922438962],\
        [ 1.191448645802597,   0.261349747400059],\
        [ 0.778004017505022,  -1.046301123377022],\
        [ 0.628873970760607,   1.103926629619643],\
        [ 1.295113890591403,  -0.479519217798997],\
        [ 1.522065175744686,   0.993476032742058],\
        [ 1.100255776045601,   0.961069161713818],\
        [-0.593243832838153,  -0.479418953496258],\
        [ 2.023196521366462,  -0.275055494808503],\
        [-0.788103134597041,  -1.090707985778480],\
        [-0.085168420896236,   1.226858390046108],\
        [ 1.691706923196703,  -1.153144804780540],\
        [ 1.989279380395157,   1.974704317386435],\
        [ 0.398799861652602,   3.051291814188982],\
        [-0.707217210772927,   0.185505264874794],\
        [ 0.697550136765320,   0.222287208720035],\
        [ 2.186126058382323,  -0.327829143438683],\
        [ 1.368068331060010,   1.708138258453435],\
        [ 0.883049126818189,  -1.334269372314072],\
        [ 1.737643116893527,   0.618452933813739],\
        [ 2.002228743955222,   0.103381966018445],\
        [-0.202638622737115,   0.495024938090909],\
        [ 0.543309203560769,  -0.802120609128192],\
        [-1.796161599703804,  -0.054795478648902],\
        [ 1.460693782000059,   0.750052171180825],\
        [ 0.133277872804608,  -1.154891068006907],\
        [ 0.203670382700157,  -0.480336687666025],\
        [-0.278985011909341,   0.030578590108392],\
        [ 2.070490237052893,   2.420782751903098],\
        [ 0.599023881366768,  -1.673208560658818],\
        [ 0.140506592147238,   0.804938444757444],\
        [-0.980799204108985,  -1.847987723222053],\
        [-0.102350006007740,  -0.822093851434857]])

    x2 = np.array([[1.160257057434194,   1.544111720606185],\
          [-0.458434595629321,   0.205667827100987],\
          [-1.053562345687376,  -0.614938261650010],\
          [-1.687901005751336,  -0.780028275457715],\
          [-0.467035854712698,   0.561692074343868],\
          [-0.703391186121452,   0.281301267639200],\
          [-1.568557779993616,  -0.629129013661319],\
          [-2.176478596101226,  -1.176211396013793],\
          [ 0.768109265900499,   1.376893437232103],\
          [-0.514772970064353,   0.474264363701950],\
          [-1.301924381487904,  -0.525179228127957],\
          [-1.312024947004566,  -0.049469442305628],\
          [-0.623417800418214,   0.226456899059445],\
          [ 0.020290591370131,   0.374055846421580],\
          [-1.002901826023476,   0.076597486786743],\
          [-2.553713136283273,  -1.731788289864902],\
          [-1.788156378743716,  -0.742460481943494],\
          [-1.119582270077321,  -0.256154464598782],\
          [-0.423084091988017,   0.395108309297119],\
          [-1.645945345460644,  -1.216319293733455],\
          [ 0.227805611684674,   0.925948003854262],\
          [-1.298719171366801,  -0.965511301629466],\
          [-0.618292817021891,   0.140045887498202],\
          [ 0.794935039731655,   1.917830760420081],\
          [-0.213709179946402,   0.617751634356751],\
          [-0.474251035850546,  -0.054854432018974],\
          [ 0.056077816960464,   1.046282980014428],\
          [ 0.887136693467512,   1.536490289895764],\
          [ 1.377161915854166,   1.764872700787871],\
          [-0.901195709427863,  -0.340855547886558],\
          [-0.783104424735034,  -0.330927422324566],\
          [-1.507139570543989,   0.137504213149820],\
          [-0.348999111724700,   0.235931187612453],\
          [-0.367309385513174,   0.655996377722041],\
          [-0.050622309620072,   0.410969334468070],\
          [ 1.734919039047271,   2.611080177877894],\
          [-0.567413078682755,  -0.458249564234885],\
          [-0.622230797920433,   0.258401595566888],\
          [-1.642146761593230,  -1.138579130251617],\
          [-0.285298076847255,   0.085451489400687]])

    x = np.concatenate((x1,x2),axis=0)
    y = np.concatenate((-np.ones((1,n1)),np.ones((1,n2))),axis=1).T

    t1,t2 = np.meshgrid(np.arange(-4,4.1,0.1),np.arange(-4,4.1,0.1))
    t = np.array(zip(np.reshape(t1,(np.prod(t1.shape),)),np.reshape(t2,(np.prod(t2.shape),)))) # these are the test inputs
    n = t.shape[0]

    tmm = np.zeros_like(t)
    tmm[:,0] = t[:,0] - m1[0]; tmm[:,1] = t[:,1] - m1[1]
    p1 = n1*np.exp( (-np.dot(tmm,S1)*tmm/2).sum(axis=1) )

    tmm = np.zeros_like(t)
    tmm[:,0] = t[:,0] - m2[0]; tmm[:,1] = t[:,1] - m2[1]
    p1 = n1*np.exp( (-np.dot(tmm,S1)*tmm/2).sum(axis=1) )
    S2i = np.array([[10.256410256410254,-9.743589743589741],[-9.743589743589741,10.256410256410254]])
    p2 = n2*np.exp( (-np.dot(tmm,S2i)*tmm/2).sum(axis=1) ) / np.sqrt(0.0975)

    '''fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(p2/(p1+p2), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()'''

    meanfunc = [ ['means.meanConst'] ] 
    covfunc  = [ ['kernels.covSEard'] ]   
    likfunc = [ ['lik.likErf'] ]
    inffunc = [ ['inf.infEP'] ]

    hyp = hyperParameters()
    hyp.mean = np.array([-2.842117459073954])
    hyp.cov  = np.array([0.051885508906388,0.170633324977413,1.218386482861781])

    '''vargout = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, np.ones((n,1)) )
    a = vargout[0]; b = vargout[1]; c = vargout[2]; d = vargout[3]; lp = vargout[4]'''

    '''fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()'''

    u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
    u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 
    del u1, u2
    nu = u.shape[0]
    covfuncF = [['kernels.covFITC'], covfunc, u]
    inffunc = [['inf.infFITC_EP'] ]         # one could also use @infFITC_Laplace
    # vargout = min_wrapper(hyp,gp,'CG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    # hyp = vargout[0]
    '''vargout = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, np.ones((n,1)) )
    a = vargout[0]; b = vargout[1]; c = vargout[2]; d = vargout[3]; lp = vargout[4]'''

    '''fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    plt.plot(u[:,0],u[:,1],'ko', markersize=12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()'''
