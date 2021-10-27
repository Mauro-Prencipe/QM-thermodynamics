# Plot module version 1.0.0 26/10/2021

# The module defines the class plot_class to produce plots of data computed
# by functions in the main bm3_thermal_2 module  

import matplotlib.pyplot as plt
import matplotlib as mpl

class plot_class():
    """
    Defines parameters and methods for the plotting of data produced
    by functions belonging to the bm3_thermal_2 module
    """
    def __init__(self, path):
        self.fsize=10
        self.tsize=10
        self.dpi=80
        self.xlab='X'
        self.ylab='Y'
        self.style='k-'
        self.ext='jpg'
        self.name='myplot'
        self.path=path
        self.tex=False
        
    def set_param(self, dpi=0, fsize=0, tsize=0, style='', name='', ext='', tex=''):
        """
        Sets parameters relevant to the production of plots.
        
        Args:
            dpi: resolution of the saved picture (default 80)
            fsize: fontsize of labels and title (default 10)
            tsize: fontsize of ticks (default 10)
            style: style of the plot (line or point types and color; default 'k-')
            name: name of the saved file (default 'myplot')
            ext: extension (and format) of the saved file (default jpg)
            tex: if True, LaTeX fonts and format is used (default False)
        
        Note: 
            Some of these parameters affect the output from the methods 'simple' and 
            'multi' of the class, as a function of the argument 'save' passed to the
            methods themselves.  
        """
        if dpi != 0:
           self.dpi=dpi
        if fsize != 0:
           self.fsize=fsize
        if tsize != 0:
           self.tsize=tsize
        if style != '':
           self.style=style
        if name != '':
           self.name=name
        if ext != '':
           self.ext=ext
        if tex != '':
           self.tex=tex
           
    def reset_param(self):
        """
        Resets parameters to their default values
        """
        self.fsize=10
        self.tsize=10
        self.dpi=80
        self.xlab='X'
        self.ylab='Y'
        self.style='k-'
        self.ext='jpg'
        self.name='myplot'
        self.tex=False

    def simple(self, x, y, title='', style='default', xlab='default', ylab='default',\
                fsize=0, tsize=0, save=False, dpi=0, name='default', ext='default',\
                tex='default'):
        """
        Plots a single (x, y) curve.
        
        Args:
            x: independent variable (array of data)
            y: dependent variable (array of data)
            title: plot title (default no title '')
            style: style of the plot (default self.style)
            xlab: X axis label (default self.xlab)
            ylab: Y axis label (default self.ylab)
            fsize: label and title size (default self.fsize)
            tsize: tick size (default self.tsize)
            save: if True, the plot will be saved on a file (default False)
            dpi: resolution of the saved file (default self.dpi)
            name: name of the saved file (default self.name)
            ext: extension (and format) of the saved file (default self.ext)
            tex: if True, tex fonts and format are used
        
        Note: 
            The arguments dpi, name and ext are relevant only if save is True
        
        Examples:
            >>> plot.simple(x,y, style='r-', xlab='T (K)', ylab='Alpha (K^-1)', fsize=12)
            
            >>> plot.set_param(fsize=12, style='r-')
            >>> plot.simple(x,y, xlab='T (K)', ylab='alpha (K^-1)')
        """
        if fsize == 0:
           fsize=self.fsize
        if tsize == 0:
           tsize=self.tsize
        if xlab == 'default':
           xlab=self.xlab
        if ylab == 'default':
           ylab=self.ylab    
        if style == 'default':
           style=self.style
        if name == 'default':
           name=self.name
        if ext == 'default':
           ext=self.ext
        if tex == 'default':
           mpl.rc('text', usetex=self.tex)
        else:
           mpl.rc('text', usetex=tex)
    
        plt.figure()
        plt.plot(x,y,style)
        plt.xlabel(xlab, fontsize=fsize)
        plt.ylabel(ylab, fontsize=fsize)
        plt.xticks(fontsize=tsize)
        plt.yticks(fontsize=tsize)
        if title != '':
            plt.suptitle(title, fontsize=fsize)
            
        if save:
           name=self.path+'/'+name+'.'+ext
           plt.savefig(name, dpi=self.dpi, bbox_inches='tight')
           
        plt.show()
        mpl.rc('text', usetex=self.tex)
        
    def multi(self, x, y, sl, lgl, title='', xlab='default', ylab='default',\
                fsize=0, tsize=0, save=False, dpi=0, name='default', ext='default',\
                tex='default'):
        """
        Plot multiple curves.
        
        Args:
            x: array of independent variable arrays
            y: array of dependent variable arrays
            sl: array of styles (one for each curve)
            lgl: array of curve labels (for the legend)
            
        Note:
            See the documentation for the 'simple' method for information 
            about the other keyword arguments.
        
        Example:
            >>> x=np.linspace(-2, 2, 24)
            >>> y1=x**2
            >>> y2=x**3
            >>> x=[x,x]
            >>> y=[y1,y2]
            >>> style=['k-', 'r+']
            >>> label=[r'Y=X$^2$', r'Y=X$^3$']
            >>> plot.multi(x,y, style, label, tex=True)
        """
    
        if fsize == 0:
           fsize=self.fsize
        if tsize == 0:
           tsize=self.tsize
        if xlab == 'default':
           xlab=self.xlab
        if ylab == 'default':
           ylab=self.ylab    
        if name == 'default':
           name=self.name
        if ext == 'default':
           ext=self.ext
        if tex == 'default':
           mpl.rc('text', usetex=self.tex)
        else:
           mpl.rc('text', usetex=tex)
    
        plt.figure()
        for ix, iy, isl, ilgl in zip(x,y,sl,lgl):
            plt.plot(ix,iy,isl, label=ilgl)
        
        plt.xlabel(xlab, fontsize=fsize)
        plt.ylabel(ylab, fontsize=fsize)
        plt.xticks(fontsize=tsize)
        plt.yticks(fontsize=tsize)
        plt.legend(frameon=False, prop={'size': fsize})
        
        if title != '':
           plt.suptitle(title, fontsize=fsize)
            
        if save:
           name=self.path+'/'+name+'.'+ext
           plt.savefig(name, dpi=self.dpi, bbox_inches='tight')
           
        plt.show()
        mpl.rc('text', usetex=self.tex)    
