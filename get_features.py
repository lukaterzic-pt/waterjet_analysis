import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.io as pio

## Functions
  

def gaus_pdf(x,µ,sig,A_var,A_offset):
   "Normaldistribution"
   return A_var*(1/(sig*np.sqrt(2*np.pi)))*(np.exp(-0.5*pow(((x-µ)/sig),2))) + A_offset

# def adapted_gaus_func(x,µ,sig,depth,A_offset):
#    "Normaldistribution"
#    return depth*(np.exp(-0.5*pow(((x-µ)/sig),2))) + A_offset  


class Feature_detektion:
    """A class for contour detection and parameter extraction from a given distribution image.

    This class provides functionality for detecting contours in an image, fitting a user-defined function
    to the detected curve, and extracting parameter values from the fitted curve.

    Attributes:
    
        image (numpy.ndarray):  ->  The input image.
        edge (numpy.ndarray):   ->  The edge-detected representation of the image.
        param (numpy.ndarray):  ->  Fitted parameters of the user-defined function.
        pcov (numpy.ndarray):   ->  Covariance matrix of the fitted parameters.
        threshold1 (int):       ->  First threshold for edge detection.
        threshold2 (int):       ->  Second threshold for edge detection.
        function (callable):    ->  User-defined function for curve fitting.
        x (numpy.ndarray):      ->  Independent variable values for the curve.
        diff (numpy.ndarray):   ->  Difference between the fitted curve and original data.
        offset (int):           ->  Offset value used for curve alignment.


    Methods:
    
        __init__(self, image_name):
            Initializes the class with an input image file.

        image2func(self):
            Extracts an funktion from the given image.

        setFunc(self, function):
            Allows setting a custom function for curve fitting.

        fitting(self):
            Performs curve fitting.

        getResult(self, b_showPlot=False):
            Retrieves the fitted curve.

        getDiff(self, b_showPlot=False):
            Computes the difference between the fitted curve and the original data.
    """
    
    name = ""
    image_path = ""
    raw_image = np.zeros(1)
    image = np.zeros(1)
    edge = np.zeros(1)
    threshold1 = 100
    threshold2 = 200
    median_blur_kernelSize = 5
    x = np.array([])
    y = np.array([])
    function = []
    features = []
    diff = []
    offset = 0
    dataType = None
    optimization_list = []
    scale = 1
    min_area = 10
    heigth = 0
    results = []
    mhu = 0
    sig = 0
    depth = 0
    off = 0
    param = 0
    final_residuals = []
    result = []
    residual_list = []
    
    def __init__(self,name):
        self.name = name

    def set_image(self,image_var):
        "Check Input"
        if isinstance(image_var, str):
            if os.path.isfile(image_var):
                self.image_path = image_var
                self.image = cv2.imread(self.image_path,cv2.IMREAD_GRAYSCALE)
                
                while len(self.y) == 0:
                    self.min_area -= 1
                    self.image_prozessing()
                    self.convert_image() 
                    if self.min_area <= 0:
                        break
            else:
                TypeError("Path of image: '" + image_var + "' not found!")
        elif isinstance(image_var, np.ndarray):
            self.image = image_var
            self.convert_image()
        else: 
            TypeError("Function '" + function + "' not found!")
        
        self.x = self.scaling(self.x,self.scale)
        self.y = self.scaling(self.y,self.scale)
        self.offset = max(self.y)  
        self.y = [item - self.offset for item in self.y]
        

        self.heigth = 0
        for h in self.image:
            self.heigth += 1

    def image_prozessing(self):
        self.image = cv2.medianBlur(self.image, self.median_blur_kernelSize)
    
    def convert_image(self):
        "Extracts a function from the given image."
        self.x = []
        self.y = []
        self.edge = cv2.Canny(self.image,self.threshold1,self.threshold2)
        # contours, _ = cv2.findContours(self.edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]
        # self.edge = np.zeros_like(self.edge)
        # cv2.drawContours(self.edge, large_contours, -1, 255, thickness=cv2.FILLED)
        
        for x in range(0,len(self.edge)):
            for y in range(0,len(self.edge[x])):
                if self.edge[x][y] < 127:
                    self.edge[x][y] = 0
                else:
                    self.edge[x][y] = 1   

        new_len = []
        orig_len = []
        list_y = []
        for x in range(0,len(self.edge[x])):
            orig_len.append(x)
            b_firstTouch = False
            for y in range(0,len(self.edge)):
                if (self.edge[y][x] & (not b_firstTouch)):
                    list_y.append(-y)
                    new_len.append(x)
                    b_firstTouch = True
        
        
        if len(list_y) != 0:
            self.y = np.asarray(list_y)
            self.x = np.arange(0,len(self.y),1)
            self.dataType = "image"
        else:
            TypeError("Edge-Detection failed: No Edge was found!")

    def scaling(self,input,scale):
        return np.array([num * scale for num in input])

    def image_scaling(self,image,scale):
        return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    
    def set_scale(self,scale):
        
        # im_scale = self.image[0:200,1500:-1]
        # pixel_count = 0

        # plt.imshow(im_scale)
        # plt.show()
        
        # for i in np.transpose(im_scale):
        #     for j in i:
        #         if j <= 10:
        #             pixel_count += 1
        #             break
                
        # self.scale = scale_length/pixel_count

        self.scale = scale


    def set_dataset(self,x_data,y_data):
        
        if len(x_data) == len(y_data):
            self.x = x_data
            self.y = y_data
            self.dataType = "dataset"
        else:
            raise TypeError("Both input arrays must be the same length!")

    def set_function(self,function):
        "Allows setting a custom function or an existing."
        if callable(function):
            self.function = function
        elif isinstance(function,str):
            if function == 'gaus':
                self.function = gaus_pdf
            else: raise TypeError("Function '" + function + "' not found!")
        else: raise TypeError("Input must be an 'callable' or an 'str'!")

    def prozess(self): 
        "Performs curve fitting."   
        if len(self.x) == 0:
            raise TypeError("Dataset error: X values are empty!")
        elif len(self.y) == 0:
            raise TypeError("Dataset error: Y values are empty!")
        else:
            self.get_features_minimize()
            self.diff = abs(self.function(self.x,*self.features) - self.y)

    def cost_function(self,params):
        _,sig,_,_ = params
        self.optimization_list.append(params)
        if sig <= 0.1:
            res = 1e15  # Bound for sig > 0
        else:
            residuals = self.y - self.function(self.x,*params)
            self.final_residuals = residuals
            res = np.sum(residuals**2)
        self.residual_list.append(res)

        return res
        
    def get_features_minimize(self):
        init = [self.x[np.argmin(self.y)],1,1,self.offset]
        bounds = [(None,None),(1e-8,None),(None,None),(None,None)]
        self.results = minimize(self.cost_function, init, bounds=bounds,method='SLSQP')
        self.rsme = np.sqrt(np.mean(self.final_residuals**2))
        self.param = self.results.x
        self.mhu = self.param[0]
        self.sig = self.param[1]
        self.depth = self.param[2] * (1/(self.param[1]*np.sqrt(2*np.pi)))
        self.off = self.param[3]
        self.features = self.mhu, self.sig, self.depth, self.off

    def get_curve(self):
        init = [self.x[np.argmin(self.y)],1,1,self.offset]
        popt,_ = curve_fit(self.function,self.x,self.y,p0=init)
        self.param = popt
        self.mhu = self.param[0]
        self.sig = self.param[1]
        self.depth = self.param[2] * (1/(self.param[1]*np.sqrt(2*np.pi)))
        self.off = self.param[3]
        self.features = self.mhu, self.sig, self.depth, self.off

    def get_result(self):
        "Retrieves the fitted curve."
        return self.mhu,self.sig,self.depth
    
    def show_result(self):
        "Retrieves the fitted curve."
        print(self.name + ":  [ µ = "+ str(round(self.mhu,3)) + "; sig = "+ str(round(self.sig,3)) +"; depth = " + str(round(self.depth,3)) + "; offset = " + str(round(self.offset,3)) + " ]")
            
    def get_optimization_list(self):
        return self.optimization_list
    
    def show_plot(self,figsize=(15, 5)):
        
        plt.figure(figsize=figsize)
        if self.dataType:
            if self.dataType == "image":
                show_image = self.image_scaling(self.image,self.scale)
                plt.imshow(show_image)
                if False:
                    plt.plot(self.x,-(self.y+self.offset),'r--',label='Kontur')
                else:
                    plt.plot(self.x,-(self.function(self.x,*self.param)+self.offset),'r--',label='Angepasste Funktion')
                plt.axhline(y=-self.offset-self.off, color='#ff9288', linestyle='--')
                plt.plot([self.mhu, self.mhu], [-self.depth-self.offset, -self.offset-self.off], linestyle='--', color='#ff5546')
                plt.text(self.mhu, -self.offset, 'µ', horizontalalignment='center')
                plt.text(self.mhu+10, -self.depth*2.8, 'Tiefe = ' + str(round(-self.depth,2))  + "µm", horizontalalignment='left')
            if self.dataType == "dataset":
                plt.axis('equal')
                plt.grid('minor')
                if self.sig > 0:
                    plt.xlim([self.mhu-(3.5*self.sig), self.mhu+(3.5*self.sig)])
                plt.plot(self.x,self.y,label='Profil')
                plt.plot(self.x,self.function(self.x,*self.param),'r--',label='Angepasste Funktion')
                plt.legend()
            
            plt.title("[sig = "+ str(round(self.sig,3)) +"µm; depth = " + str(round(self.depth,3)) + "µm]") 
            plt.xlabel('Querschnitt (µm)')
            plt.ylabel('Probentiefe (µm)')
            plt.show()
            
    def get_plot(self):
        
        f=[]   
        f = plt.figure(figsize=(8, 6))
        if self.dataType:
            if self.dataType == "image":
                show_image = self.image_scaling(self.image,self.scale)
                plt.imshow(show_image)
                plt.plot(self.x,-(self.function(self.x,*self.param)+self.offset),'r--',label='Angepasste Funktion')
                plt.axhline(y=-self.offset-self.off, color='#ff9288', linestyle='--')
                plt.plot([self.mhu, self.mhu], [-self.depth-self.offset, -self.offset-self.off], linestyle='--', color='#ff5546')
                plt.text(self.mhu, -self.offset, 'µ', horizontalalignment='center')
                plt.text(self.mhu+10, -self.depth*2.8, 'Tiefe = ' + str(round(-self.depth,2))  + "µm", horizontalalignment='left')
            if self.dataType == "dataset":
                plt.axis('equal')
                plt.grid('minor')
                plt.xlim([self.mhu-(3.5*self.sig), self.mhu+(3.5*self.sig)])
                plt.plot(self.x,self.y,label='Profil')
                plt.plot(self.x,self.function(self.x,*self.param),'r--',label='Angepasste Funktion')
                plt.legend()
            
            plt.title("[sig = "+ str(round(self.sig,3)) +"µm; depth = " + str(round(self.depth,3)) + "µm]") 
            plt.xlabel('Querschnitt (µm)')
            plt.ylabel('Probentiefe (µm)')
        return f

    
    def get_IO_differenze(self,b_showPlot):
        "Computes the difference between the fitted curve and the original data."  
        if b_showPlot:
            plt.plot(self.x,self.diff)
            plt.show()
        return self.diff
    

def show_on_browser(profile,markersize=None,marker_colorscale=None):
    
    if markersize is None:
        markersize = 3
    if marker_colorscale is None:
        marker_colorscale = 'Viridis'
    
    try:
        x = profile[:,0]
        y = profile[:,1]
        z = profile[:,2]
    except:
        x = profile[0]
        y = profile[1]
        z = profile[2]
    
    
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                            marker=dict(size=markersize, color=z, colorscale=marker_colorscale))])

    fig.update_layout(scene=dict(
                        xaxis_title='X Achse',
                        yaxis_title='Y Achse',
                        zaxis_title='Z Achse'),
                    margin=dict(l=0, r=0, b=0, t=0))


    pio.show(fig)
        
def get_notch(profile_3d,xlim=None, ylim=None, zlim=None):
    x = profile_3d[0]
    y = profile_3d[1]
    z = profile_3d[2]
    x_new = np.array([])
    y_new = np.array([])
    z_new = np.array([])
    
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]
    if zlim is None:
        zlim = [np.min(z), np.max(z)]
        
    for p in range(1,len(x)): 
        if x[p] > xlim[0] and x[p] < xlim[1] and y[p] > ylim[0] and y[p] < ylim[1] and z[p] > zlim[0] and z[p] < zlim[1]:
            x_new = np.append(x_new,x[p])
            y_new = np.append(y_new,y[p])
            z_new = np.append(z_new,z[p])
    return [x_new,y_new,z_new]

def get_kontur(notch_3d, length=None, start=None, end=None):
    x = notch_3d[0]
    y = notch_3d[1]
    z = notch_3d[2]
    x_new = np.array([])
    y_new = np.array([])
    new_slice = []
    
    # Konselation wenn ende und länge existiert noch nicht richtig!!!
    if start is None:
        start = np.min(y)
        
    if length is None:
        end = np.max(y)
    else:
        end = start + length
        
    if end is None:
        end = np.max(y)
    
    y_old = start
    p_old = np.where(y == start)[0][0]
    print(p_old)
    count = 0
    for p in range(1,len(y)):
        if y[p] > y_old:
            if y[p] > end: break
            count = count+1
            new_slice.append([count,y[p],x[p_old:p],z[p_old:p]])
            p_old = p
            y_old = y[p_old]

    return new_slice


    
