import xarray as xr
import rioxarray
import salem
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.patches as mpatches

from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error

# generate plot of shaded ROI

# project inputs: xy=[-47, 59.45], width=5, height=3
def shaded_ROI(min_long, min_lat, w, h): 
    '''
    Generate plot of shaded ROI

    Input: 
        min_long: int64
            Minimum Longitude Value (note: neg. value if W coord.)
        max_long: int64
            Minimum Longitude Value
        w: int64
            width to expand bounding box (bb) with
        h: int64
            height to expand bb with        
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=-60, central_latitude=50))


    ax.add_patch(mpatches.Rectangle(xy=[min_long, min_lat], width=w, height=h,
                                    facecolor='red',
                                    alpha=0.2,
                                    transform=ccrs.PlateCarree())
                 )

    ax.coastlines(resolution='auto', color='k')
    ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    ax.coastlines()
    ax.stock_img()

    plt.show()


def plot_prediction(X, Y, model_name, n_toplot=10**10):
    '''
    Density scatter plot of the results 
    '''
    
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    
    y = X.reshape(-1)[idxs[:n_toplot]]
    y_star = Y.reshape(-1)[idxs[:n_toplot]]

    XY = np.vstack([y, y_star])
    Z = gaussian_kde(XY)(XY)
    
    idx = z.argsort()
    y_plt, ann_plt, z = y[idx], y_star[idx], Z[idx]
    
    plt.figure(figsize=(8,8))
    plt.title(model_name, fontsize=17)
    plt.ylabel('Predicted SMB (m.w.e)', fontsize=16)
    plt.xlabel('Known SMB (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=20)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc) 
    lineS = -2.5
    lineE = 1.5
    plt.plot([lineS, lineE], [lineS, lineE], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineS, lineE)
    plt.ylim(lineS, lineE)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.2f$' % (mean_squared_error(y, y_star), ),
    r'$R^2=%.2f$' % (r2_score(y, y_star), )))
    props = dict(boxstyle='round', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.show()










