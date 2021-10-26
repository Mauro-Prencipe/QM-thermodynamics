# Plot module version 1.0.0 23/10/2021

import matplotlib.pyplot as plt
import matplotlib as mpl

path_file=open('path_file.dat', 'r')
path=path_file.readline().rstrip()
path_file.close()

class plot_class():
    def __init__(self):
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

    def simple(self, x, y, title='', style='default', xlab='default', ylab='default',\
                fsize=0, tsize=0, save=False, dpi=0, name='default', ext='default',\
                tex='default'):
    
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
        plt.legend(frameon=False)
        
        if title != '':
           plt.suptitle(title, fontsize=fsize)
            
        if save:
           name=self.path+'/'+name+'.'+ext
           plt.savefig(name, dpi=self.dpi, bbox_inches='tight')
           
        plt.show()
        mpl.rc('text', usetex=self.tex)    
