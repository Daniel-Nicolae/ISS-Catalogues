import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def deg_to_rad(x):

    return x*np.pi/180

def standard_dev(a):
    mean = sum(a)/len(a)
    sd = 0
    for y in a:
        sd+=(y-mean)*(y-mean)
    sd = np.sqrt(sd/(len(a)-1))

    return mean, sd

def moving_avg(x):

    if(len(x)<3):
        return x
    xn = [(x[0]+x[1])/2]
    i=1
    while i<len(x)-1:
        xn.append((x[i-1]+x[i]+x[i+1])/3)
        i+=1
    xn.append((x[-1]+x[-2])/2)
    return xn

def equ_to_gal(a, d):
    
    a = deg_to_rad(a)
    d = deg_to_rad(d)

    dp = deg_to_rad(27.13)
    ap = deg_to_rad(12.8567*15)
    lcp = deg_to_rad(122.917)
    b = np.arcsin(np.sin(dp)*np.sin(d)+np.cos(dp)*np.cos(d)*np.cos(a-ap))
    l = lcp - np.arcsin(np.cos(d)*np.sin(a-ap)/np.cos(b))

    return b, l


def Metropolis_Hastings(f, x0, sd, min, max, N=10000, plot=0, stat=0):

    x = [x0]
    i=1
    while i<N:
        x_current = x[-1]
        x_proposed = np.random.normal(x_current, sd)

        if x_proposed <= min or x_proposed >=max:
            A=-1
        else:
            A = f(x_proposed)/f(x_current)
        y=random.uniform(0, 1)
        if y < A:
            x.append(x_proposed)
            i+=1
        elif A>=0:
            x.append(x_current)
            i+=1
    plt.clf()
    n, bins, p, =plt.hist(x, bins=40, histtype='step')
    n = moving_avg(n)
    n = moving_avg(n)
    bins=bins[:-1]
    if plot:
        plt.plot(bins, n)
        plt.show()
    else:
        plt.clf()
    x2 = np.array(x)

    if stat:
        return x2, n, bins
    else:
        return x2



def statistical_errors(f, steps=200):
    
    new_data_matrix = []
    new_curves_matrix = []

    plt.clf()

    for s in range(steps):
        start = np.random.uniform(0, 700)
        row_data, row_curve, bins = Metropolis_Hastings(f, start, sd=100, min = 0, max=1500, N=num, stat =1)

        row_data.sort()
        new_data_matrix.append(row_data)
        new_curves_matrix.append(row_curve)
    
    for s in range(steps):
        plt.plot(bins, new_curves_matrix[s], ":")

    new_data_matrix = np.array(new_data_matrix)
    new_curves_matrix = np.array(new_curves_matrix)

    np.savetxt("massess.csv", new_data_matrix, delimiter=',')

    max_curve = []
    min_curve = []
    for i in range(len(new_curves_matrix[0])):
        y_max = max(new_curves_matrix[:, i])
        y_min = min(new_curves_matrix[:, i])
        y_mid = (y_max+y_min)/2

        y_max = y_mid + (y_max-y_mid)/np.sqrt(2)
        y_min = y_mid - (y_mid-y_min)/np.sqrt(2)

        max_curve.append(y_max)
        min_curve.append(y_min)

    plt.plot(bins, max_curve, linewidth=2)   
    plt.plot(bins, min_curve, linewidth=2)

    plt.title("Mass distribution curves")
    plt.xlabel("Mass ($M_\odot$)")
    plt.ylabel("Histogram counts")
    plt.grid()

    

    results = []
    for i in range(len(new_data_matrix[0])):
        mean, sd = standard_dev(new_data_matrix[:, i])
        results.append((mean, sd))

    random.shuffle(results)
    values = []
    errors = []
    for i in results:
        values.append(i[0])
        errors.append(i[1])

    plt.show()
    return values, errors



def fit_data(data, degree=10, name="", label = "", plot=0):

    # Converting csv data to Numpy array for easier computing
    data_arr = np.array(data)
    data_arr = data_arr.astype('float32')

    # Creating a histogram and extracting the y values.
    n, bins, p= plt.hist(data_arr, bins=70, histtype='step', label="hist")
    bins=bins[:-1]
    plt.clf()

    # Plotting an unnormalised distribution function  
    plt.plot(bins, n, label="n")


    # Fitting the distribution function with a polynomial
    coef= np.polyfit(bins, n, degree)
    poly = np.poly1d(coef)

    
    if plot:
        plt.plot(bins, poly(bins), '--')
        plt.title(name)
        plt.xlabel(label)
        plt.ylabel("Histogram counts")
        plt.grid()
        plt.show()

    return poly

# Importing the csv catalogue
data = pd.read_csv("catalogue2.csv")

# Show initial map of data points in galactic coordinates
plt.scatter(data['GLON'], data['GLAT'], s=4)
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")
#plt.show()


# Establish the limits of the galactic plane gap
lat_gap_lower = -12
lat_gap_upper = 14.5

# Counting the points which are already in the gap and those in the neighbouring bands
middle_data=0
upper_data = 0
lower_data = 0
for l in data['GLAT']:
    if l > lat_gap_lower and l < 0:
        middle_data+=1
    if l < lat_gap_upper and l>0:
        middle_data+=1
    if l>lat_gap_upper and l < 2*lat_gap_upper:
        upper_data+=1
    if l < lat_gap_lower and l > 2*lat_gap_lower:
        lower_data+=1


lat = np.array(data['GLAT'])
poly = fit_data(lat, 10, name = "Galactic Latitude", label="Latitude (degrees)",  plot=0)

# Latitude generation - using a uniform distribution
num = upper_data+lower_data-middle_data
new_lat = np.random.uniform(lat_gap_lower, lat_gap_upper, size=num)
lat_total = np.concatenate((lat, new_lat))

# Check final latitude histogram
poly = fit_data(lat_total, 10, name = "Galactic Latitude", label="Latitude (degrees)", plot=1)

# Longitude generation
poly = fit_data(data['GLON'],10,  name="Galactic Longitude",label="Longitude (degrees)", plot=0)
new_lon = Metropolis_Hastings(poly, 180, sd=20, min = 0, max=360, N=num)

# Plot new data points map
lon = np.array(data['GLON'])
lon_total = np.concatenate((lon, new_lon))
plt.clf()
plt.scatter(lon_total, lat_total, s=4)
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")
plt.show()

poly = fit_data(lat_total, 10, name = "Galactic Latitude", label="Latitude (degrees)", plot=1)

# Redshift generation
poly = fit_data(data['z'],10,  name="Redshift",label="z", plot=0)
new_z = Metropolis_Hastings(poly, 0.01, sd=0.001, min = 0, max=0.025, N=num)
z=np.array(data['z'])
z_total=np.concatenate((z, new_z))

# Mass generation
poly = fit_data(data['Mbh'],9,  name="Mass",label="Mass ($M_\odot$)", plot=0)
#new_M = Metropolis_Hastings(poly, 600, sd=150, min = 0, max=1500, N=num, plot=1)
new_M , err = statistical_errors(poly, steps=1)

# Export csv with new data
new_data = {'Redshift': new_z, 'GLAT': new_lat, 'GLON': new_lon, 'Mass': new_M, "Mass error": err}
new_csv = pd.DataFrame(new_data)
new_csv.to_csv('new_data.csv')

# Convert galactic coordinates into cartesian for 3d plot - first all on unit sphere
x_3d = np.cos(deg_to_rad(lat))*np.cos(deg_to_rad(lon))
y_3d = np.cos(deg_to_rad(lat))*np.sin(deg_to_rad(lon))
z_3d = np.sin(deg_to_rad(lat))

x_3d_new = np.cos(deg_to_rad(new_lat))*np.cos(deg_to_rad(new_lon))
y_3d_new = np.cos(deg_to_rad(new_lat))*np.sin(deg_to_rad(new_lon))
z_3d_new = np.sin(deg_to_rad(new_lat))

# 3D plot of data points projected on unit sphere
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_3d, y_3d, z_3d, s=2)
ax.scatter(x_3d_new, y_3d_new, z_3d_new, s=2)
#plt.show()
plt.clf()

# Convert redshift into distance in m
dist = z*(3*10**5)/70
dist_new = new_z*(3*10**5)/70

# Adjust cartesian coordinates for distance
x_3d*=dist
y_3d*=dist
z_3d*=dist

x_3d_new*=dist_new
y_3d_new*=dist_new
z_3d_new*=dist_new

# 3D plot of data points with distance
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_3d, y_3d, z_3d, s=2)
ax.scatter(x_3d_new, y_3d_new, z_3d_new, s=1)
#plt.show()

